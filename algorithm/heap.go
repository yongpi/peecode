package main

import "fmt"

func maxK(num []int, k int) int {
	initHeap(num)
	size := len(num)

	for i := 1; i < k; i++ {
		num[0], num[size-1] = num[size-1], num[0]
		size--
		adjustHeap(num, 0, size)
	}

	return num[0]
}

func numK(num []int, k int) []int {
	initHeap(num)
	size := len(num)
	ans := make([]int, 0)
	for i := 0; i < k; i++ {
		ans = append(ans, num[0])

		num[0], num[size-1] = num[size-1], num[0]
		size--
		adjustHeap(num, 0, size)
	}

	return ans
}

func initHeap(num []int) {
	si := len(num)/2 - 1
	for i := si; i >= 0; i-- {
		adjustHeap(num, i, len(num))
	}
}

func adjustHeap(num []int, index, size int) {
	ls, rs, target := index*2+1, index*2+2, index

	if ls < size && num[ls] > num[target] {
		target = ls
	}
	if rs < size && num[rs] > num[target] {
		target = rs
	}

	if target != index {
		num[target], num[index] = num[index], num[target]
		adjustHeap(num, target, size)
	}
}

func main() {
	nums := []int{2, 3, 1, 4, 5, 6, 9, 10}
	fmt.Println(numK(nums, 3))
}
