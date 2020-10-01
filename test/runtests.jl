using NonsmoothOptim
using Test

@testset "NonsmoothOptim.jl" begin
  @testset "Subgradient" begin
    @testset "1D: abs(x)" begin
      f = (x) -> abs(x[1])
      g = x -> [(x[1] < 0) ? -1.0 : 1.0]

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
    end
  end
  
  @testset "Projected subgradient" begin
  end
  
  @testset "Bundle" begin
  end
end
