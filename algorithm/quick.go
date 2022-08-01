package main

import (
	"fmt"
)

func Quick(list []int) []int {
	Sort(list, 0, len(list)-1)
	return list
}

func Sort(list []int, lo, hi int) {
	if lo >= hi {
		return
	}
	p := Position(list, lo, hi)
	Sort(list, lo, p-1)
	Sort(list, p+1, hi)
}

func Position(list []int, lo, hi int) int {
	cv := list[lo]
	i := lo + 1
	j := hi

	for {
		for ; list[i] <= cv; i++ {
			if i == hi {
				break
			}
		}
		for ; list[j] > cv; j-- {
			if j == lo {
				break
			}
		}
		if i >= j {
			break
		}
		ExChange(list, i, j)
	}

	ExChange(list, lo, j)
	return j
}

func ExChange(list []int, lo, hi int) {
	lv := list[lo]
	list[lo] = list[hi]
	list[hi] = lv
}

func main() {
	fmt.Println(Quick([]int{2, 2, 3, 1, 5, 2, 4}))
}

func Position2(list []int, lo, hi int) int {
	src := list[lo]
	i := lo + 1
	j := hi

	for {
		for ; src > list[i]; i++ {
			if i == hi {
				break
			}
		}

		for ; src < list[j]; j-- {
			if j == lo {
				break
			}
		}

		if i >= j {
			break
		}

		ExChange(list, i, j)
	}

	ExChange(list, lo, j)
	return j
}

func Sort2(list []int, lo, hi int) {
	if lo >= hi {
		return
	}
	p := Position2(list, lo, hi)
	Sort2(list, lo, p-1)
	Sort2(list, p+1, hi)
}
