# ── Thread-environment wrappers for a streaming pass ─────────────────────────
#
# A pass sizes its own group concurrency (`ScanStream.ntasks`); these bound the
# nested thread pools the group tasks reach into, so the two multiply out to
# roughly the core count instead of oversubscribing it.

# Run `f` with BLAS pinned to a single thread, restoring the prior setting after.
# The fringe solve tasks over scan groups (see `_scheduled_map`); if BLAS also
# spawns threads, every task's WLS/QR solve fans out `BLAS.get_num_threads()`
# threads, so `ntasks × blas_threads` (e.g. 8 × 8 = 64) oversubscribe the cores
# and contend — threads sit "runnable" while only ~1 core makes progress. One
# BLAS thread per task is the correct split when the parallelism is across tasks.
function _with_single_blas_thread(f)
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(old)
    end
end

# Run `f` with FFTW using `n` threads per transform, restoring 1 (FFTW's default)
# after. The fringe SEARCH is ~96% FFT, but the memory cap pins group-parallelism
# (`ntasks`) well below the core count on a big file — leaving cores idle DURING
# the search. Giving each FFT `n = nthreads ÷ ntasks` threads uses them
# (`ntasks × n ≈ nthreads`), so the otherwise-idle cores accelerate the transforms
# instead of sitting out the bottleneck phase. Plans are built lazily inside the
# threaded passes, so this must wrap them to take effect.
function _with_fft_threads(f, n::Int)
    FFTW.set_num_threads(max(1, n))
    try
        return f()
    finally
        FFTW.set_num_threads(1)
    end
end

# Run `f` with the bulk reader decoding each leaf over `n` tasks, restoring 1
# after. Decode (byte-swap + complex repack + pol permute) is CPU-bound, so like
# the FFT it can use the cores the memory cap leaves idle. Nested under the group
# group tasks, but the task scheduler bounds live parallelism, so it composes.
function _with_decode_threads(f, n::Int)
    old = UVData._DECODE_NTASKS[]
    UVData._DECODE_NTASKS[] = max(1, n)
    try
        return f()
    finally
        UVData._DECODE_NTASKS[] = old
    end
end
