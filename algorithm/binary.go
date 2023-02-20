package main

import "fmt"

func Binary(list []int, src int) int {
	lo := 0
	hi := len(list) - 1

	for lo < hi {
		mid := lo + (hi-lo)/2
		if src > list[mid] {
			lo = mid + 1
		}
		if src < list[mid] {
			hi = mid - 1
		}
		if src == list[mid] {
			return mid
		}
	}

	if list[lo] != src {
		return -1
	}
	return lo
}

func main() {
	fmt.Println(binary([]int{1, 3, 4, 5, 8, 9}, 6))
}

func Binary2(list []int, src int) int {
	lo := 0
	hi := len(list) - 1

	for lo < hi {
		mid := lo + (hi-lo)/2
		if src > list[mid] {
			lo = mid + 1
		}
		if src < list[mid] {
			hi = mid - 1
		}
		if src == list[mid] {
			return mid
		}
	}

	if list[lo] == src {
		return lo
	}

	return -1
}

func binary(num []int, target int) int {
	li := 0
	ri := len(num) - 1

	var ans int
	for li <= ri {
		mid := li + (ri-li)/2
		if num[mid] >= target {
			ans = mid
			ri = mid - 1
		} else {
			li = mid + 1
		}
	}

	return ans
}
