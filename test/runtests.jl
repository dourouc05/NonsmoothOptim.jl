using NonsmoothOptim
using Test

@testset "NonsmoothOptim.jl" begin
  @testset "Subgradient" begin
    @testset "1D: abs(x)" begin
      f = (x) -> abs(x[1])
      g = (x) -> [(x[1] < 0) ? -1.0 : 1.0]

      @testset "Minimise 1D" begin
        p = UnconstrainedNonSmoothProblem(f, g, 1, Minimise)
        a = SubgradientMethod(ConstantStepSize(1.0), 2)

        x0 = [10.0]
        x1 = solve(p, a, x0)
        @test length(x1) == 1
        @test x1[1] ≈ 8.0
      end

      @testset "Minimise 1D (start at optimum)" begin
        p = UnconstrainedNonSmoothProblem(f, g, 1, Minimise)
        a = SubgradientMethod(ConstantStepSize(1.0), 2)

        x0 = [0.0]
        x1 = solve(p, a, x0)
        @test length(x1) == 1
        @test x1[1] ≈ x0[1]
      end

      @testset "Maximise 1D" begin
        p = UnconstrainedNonSmoothProblem(f, g, 1, Maximise)
        a = SubgradientMethod(ConstantStepSize(1.0), 2)

        x0 = [10.0]
        x1 = solve(p, a, x0)
        @test length(x1) == 1
        @test x1[1] ≈ 12.0
      end

      @testset "Maximise 1D (2)" begin
        p = UnconstrainedNonSmoothProblem(f, g, 1, Maximise)
        a = SubgradientMethod(ConstantStepSize(1.0), 2)

        x0 = [-10.0]
        x1 = solve(p, a, x0)
        @test length(x1) == 1
        @test x1[1] ≈ -12.0
      end

      @testset "Info callback" begin
        cb_called = false
        cb = (k, f, x, g, f_best, x_best, δt) -> begin
          cb_called = true
        end
        p = UnconstrainedNonSmoothProblem(f, g, 1, Maximise)
        a = SubgradientMethod(ConstantStepSize(1.0), 2)
        solve(p, a, [0.0], info_callback=cb)

        @test cb_called
      end
    end
  end
  
  @testset "Projected subgradient" begin
    @testset "1D: abs(x)" begin
      f = (x) -> abs(x[1])
      g = (x) -> [(x[1] < 0) ? -1.0 : 1.0]
      proj = (x) -> [200.0]

      @testset "Minimise 1D" begin
        p = ProjectedConstrainedNonSmoothProblem(f, g, proj, 1, Minimise)
        a = ProjectedSubgradientMethod(ConstantStepSize(1.0), 2)

        x0 = [10.0] # Infeasible solution! Must be projected!
        x1 = solve(p, a, x0)
        @test length(x1) == 1
        @test x1[1] ≈ 200.0
      end
    end
  end
  
  @testset "Bundle" begin
  end
end
