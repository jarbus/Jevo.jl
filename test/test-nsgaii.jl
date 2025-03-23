@testset "NSGA-II Maximization Tests" begin
    @testset "domination" begin
        # Test 1: Simple 4-individual problem with two objectives.
        # For maximization:
        # - Row 4: (4.0, 4.0) is best and non-dominated.
        # - Row 3: (3.0, 3.0) is dominated only by row 4.
        # - Rows 1: (1.0, 2.0) and 2: (2.0, 1.0) are dominated by rows 3 and 4.
        outcomes = [
            1.0 2.0;
            2.0 1.0;
            3.0 3.0;
            4.0 4.0
        ]
        
        # When selecting 2 parents, expect rows 4 and 3.
        selected = Jevo.nsga2(outcomes, 2)
        @test 4 in selected && 3 in selected
        
        # Test 2: Request 3 parents.
        selected = Jevo.nsga2(outcomes, 3)
        @test length(selected) == 3
        @test 4 in selected && 3 in selected
        
        # Test 3: Selecting all individuals.
        selected = Jevo.nsga2(outcomes, 4)
        @test length(selected) == 4
        @test sort(selected) == [1, 2, 3, 4]
    end

    @testset "Crowding Distance Preservation Test" begin
        # This test creates a non-dominated set where the extreme solutions in each objective
        # should be preserved due to their infinite crowding distances.
        # For maximization, consider:
        # Row 1: (1.0, 5.0) and Row 5: (5.0, 1.0)
        # are extremes (best in one objective and worst in the other) and receive Inf distance.
        outcomes = [
            1.0 5.0;
            2.0 4.0;
            3.0 3.0;
            4.0 2.0;
            5.0 1.0
        ]
        
        # When selecting fewer parents than individuals, the extreme individuals (rows 1 and 5)
        # should always be chosen due to their crowding distance.
        selected = Jevo.nsga2(outcomes, 2)
        @test 1 in selected && 5 in selected

        # Also, for p=3, the extreme individuals should be included.
        selected = Jevo.nsga2(outcomes, 3)
        @test 1 in selected && 5 in selected
    end
end


