from arc_types import Grid
import dsl

INPUT_1: Grid = ((1,0,0),(0,1,0),(0,0,1))
SOLUTION_1: Grid = ((0,0,2),(0,2,0),(2,0,0))

INPUT_2: Grid = ((0,1,0),(0,1,0),(0,0,1))
SOLUTION_2: Grid = ((0,2,0),(0,2,0),(2,0,0))

def just_get_one_done(input:Grid) -> Grid:
    return dsl.replace(dsl.vmirror(input),1,2)

def main():
    assert just_get_one_done(INPUT_1)==SOLUTION_1
    assert just_get_one_done(INPUT_2)==SOLUTION_2


if __name__ == '__main__':
    main()
