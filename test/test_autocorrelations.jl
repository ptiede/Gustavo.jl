# ── Normalizing a Measurement Set by its autocorrelations ─────────────────────

using XRadio: XRadio

# A Measurement Set of antennas A1–A3 holding every baseline including each
# antenna with itself, relabeled from the synthetic builder's six cross
# baselines among four antennas. A2's receptors are stored `L, R`, so its
# feed 1 is `L`.
function _autocorrelated_ms(; antenna2 = ["A1", "A2", "A3", "A2", "A3", "A3"])
    names = ["A1", "A2", "A3", "A4"]
    ant = XRadio.Testing.antenna(names)
    parent(ant[:polarization_type])[:, 2] .= ["L", "R"]
    ms = XRadio.Testing.measurement_set(; antennas = names, antenna_xds = ant)
    parent(ms[:baseline_antenna1_name]) .= ["A1", "A1", "A1", "A2", "A2", "A3"]
    parent(ms[:baseline_antenna2_name]) .= antenna2
    return ms
end

_power(a, f, c, t) = Float32(a + 3f + c / 10 + t / 100)

function _set_autocorrelations!(ms)
    UV = Gustavo.UVData
    feeds = UV.feed_pairs(ms)
    vis = parent(ms[:visibility])
    for (bi, (a, b)) in pairs(UV.baselines(ms).pairs), p in axes(feeds, 1)
        fa, fb = feeds[p, bi]
        (a == b && fa == fb) || continue
        for c in axes(vis, 2), t in axes(vis, 4)
            vis[p, c, bi, t] = _power(a, fa, c, t)
        end
    end
    return ms
end

@testset "normalize_by_autocorrelations" begin
    UV = Gustavo.UVData

    @testset "divides each product by its feeds' autocorrelations" begin
        ms = _set_autocorrelations!(_autocorrelated_ms())
        before = copy(parent(ms[:visibility]))
        out = UV.normalize_by_autocorrelations(ms)
        @test parent(ms[:visibility]) == before
        @test !any(parent(ms[:flag]))

        feeds = UV.feed_pairs(out)
        vis, weight, flag = parent(out[:visibility]), parent(out[:weight]), parent(out[:flag])
        for (bi, (a, b)) in pairs(UV.baselines(out).pairs)
            if a == b
                @test all(view(flag, :, :, bi, :))
                @test view(vis, :, :, bi, :) == view(before, :, :, bi, :)
                continue
            end
            for p in axes(feeds, 1), c in axes(vis, 2), t in axes(vis, 4)
                fa, fb = feeds[p, bi]
                power = _power(a, fa, c, t) * _power(b, fb, c, t)
                @test vis[p, c, bi, t] ≈ 1 / sqrt(power)
                @test weight[p, c, bi, t] ≈ power
                @test !flag[p, c, bi, t]
            end
        end
        # A1–A2 `RL` relates A1's feed 1 to A2's feed 1, whose autocorrelation
        # A2 stores as `LL`.
        @test feeds[2, 2] == (1, 1)
        @test vis[2, 1, 2, 1] ≈ 1 / sqrt(_power(1, 1, 1, 1) * _power(2, 1, 1, 1))
    end

    @testset "flags instead of dividing where an autocorrelation is unusable" begin
        ms = _set_autocorrelations!(_autocorrelated_ms())
        vis = parent(ms[:visibility])
        rr = findfirst(==("RR"), UV.pol_products(ms))
        parent(ms[:flag])[rr, 3, 1, 1] = true    # A1's feed 1, channel 3, time 1
        vis[rr, 4, 1, 2] = 0                     # A1's feed 1, channel 4, time 2
        out = UV.normalize_by_autocorrelations(ms)

        feeds = UV.feed_pairs(out)
        flag = parent(out[:flag])
        for (bi, (a, b)) in pairs(UV.baselines(out).pairs), p in axes(feeds, 1)
            a == b && continue
            uses = (a == 1 && feeds[p, bi][1] == 1) || (b == 1 && feeds[p, bi][2] == 1)
            @test flag[p, 3, bi, 1] == uses
            @test flag[p, 4, bi, 2] == uses
            @test count(view(flag, p, :, bi, :)) == 2 * uses
            if uses
                @test parent(out[:visibility])[p, 3, bi, 1] == vis[p, 3, bi, 1]
                @test parent(out[:weight])[p, 3, bi, 1] == 1
            end
        end
    end

    @testset "flags every product of an antenna with no autocorrelation" begin
        ms = _set_autocorrelations!(
            _autocorrelated_ms(; antenna2 = ["A1", "A2", "A3", "A2", "A3", "A4"])
        )
        out = UV.normalize_by_autocorrelations(ms)
        flag = parent(out[:flag])
        for (bi, (a, b)) in pairs(UV.baselines(out).pairs)
            @test all(view(flag, :, :, bi, :)) == (a == b || 3 in (a, b) || 4 in (a, b))
            @test any(view(flag, :, :, bi, :)) == all(view(flag, :, :, bi, :))
        end
    end

    @testset "leaves a set without autocorrelations unchanged" begin
        ms = XRadio.Testing.measurement_set()
        @test UV.normalize_by_autocorrelations(ms) === ms
    end
end
