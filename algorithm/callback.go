package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

func numAndKLenList(n, k int) [][]int {
	var ans [][]int
	var stack []int

	var callBack func(index int)
	callBack = func(index int) {
		if len(stack) == k {
			var tmp []int
			tmp = append(tmp, stack...)
			ans = append(ans, tmp)
			return
		}

		for i := index; i <= n; i++ {
			stack = append(stack, i)
			callBack(i + 1)
			stack = stack[:len(stack)-1]
		}
	}

	callBack(1)
	return ans
}

func listAndSumTarget(num []int, target int) [][]int {
	var ans [][]int
	var stack []int

	var callBack func(target, idx int)
	callBack = func(target, idx int) {
		if idx == len(num) {
			return
		}

		if target == 0 {
			var tmp []int
			tmp = append(tmp, stack...)
			ans = append(ans, tmp)
			return
		}

		callBack(target, idx+1)

		if target >= num[idx] {
			stack = append(stack, num[idx])
			callBack(target-num[idx], idx)
			stack = stack[:len(stack)-1]
		}
	}

	callBack(target, 0)
	return ans
}

func listAndSumNotRepeatTarget(num []int, target int) [][]int {
	sort.Ints(num)
	var seq [][2]int
	var ans [][]int
	var stack []int

	for _, value := range num {
		if seq == nil || seq[len(seq)-1][0] != value {
			seq = append(seq, [2]int{value, 1})
		} else {
			seq[len(seq)-1][1]++
		}
	}

	var callBack func(target, index int)
	callBack = func(target, index int) {
		if index == len(seq) {
			return
		}
		if target == 0 {
			var tmp []int
			tmp = append(tmp, stack...)
			ans = append(ans, tmp)
			return
		}

		callBack(target, index+1)

		if target < seq[index][0] {
			return
		}

		count := min(seq[index][1], target/seq[index][0])
		for i := 1; i <= count; i++ {
			stack = append(stack, seq[index][0])
			callBack(target-i*seq[index][0], index+1)
		}
		stack = stack[:len(stack)-count]
	}

	callBack(target, 0)
	return ans
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

func discountList(num []int) [][]int {
	var ans [][]int
	var stack []int
	used := make(map[int]bool)

	var callBack func()
	callBack = func() {
		if len(stack) == len(num) {
			var tmp []int
			tmp = append(tmp, stack...)
			ans = append(ans, tmp)
			return
		}

		for i := 0; i < len(num); i++ {
			if !used[i] {
				stack = append(stack, num[i])
				used[i] = true
				callBack()
				used[i] = false
				stack = stack[:len(stack)-1]
			}
		}
	}

	callBack()
	return ans
}

func repeatList(num []int) [][]int {
	sort.Ints(num)

	var ans [][]int
	var stack []int
	used := make(map[int]bool)

	var callBack func()
	callBack = func() {
		if len(stack) == len(num) {
			var tmp []int
			tmp = append(tmp, stack...)
			ans = append(ans, tmp)
			return
		}

		for i := 0; i < len(num); i++ {
			if used[i] || (i > 0 && !used[i-1] && num[i-1] == num[i]) {
				continue
			}

			stack = append(stack, num[i])
			used[i] = true
			callBack()
			used[i] = false
			stack = stack[:len(stack)-1]
		}
	}

	callBack()
	return ans
}

func kuoHao(n int) []string {
	var ans []string
	var stack []byte

	var callBack func(open, close int)
	callBack = func(open, close int) {
		if len(stack) == 2*n {
			data := string(stack)
			ans = append(ans, data)
			return
		}

		if open < n {
			stack = append(stack, '(')
			callBack(open+1, close)
			stack = stack[:len(stack)-1]
		}
		if close < open {
			stack = append(stack, ')')
			callBack(open, close+1)
			stack = stack[:len(stack)-1]
		}
	}

	callBack(0, 0)
	return ans
}

func position(data string) [][]string {
	n := len(data)
	dp := make([][]bool, len(data))
	for i := range dp {
		dp[i] = make([]bool, len(data))
		for j := range dp[i] {
			if i >= j {
				dp[i][j] = true
			}
		}
	}

	for i := n - 1; i >= 0; i-- {
		for j := i + 1; j < n; j++ {
			dp[i][j] = dp[i+1][j-1] && data[i] == data[j]
		}
	}

	var callBack func(i int)
	var stack []string
	var ans [][]string

	callBack = func(i int) {
		if i == len(data) {
			var tmp []string
			tmp = append(tmp, stack...)
			ans = append(ans, tmp)
			return
		}

		for j := i; j < len(data); j++ {
			if dp[i][j] {
				tv := data[i : j+1]
				stack = append(stack, tv)
				callBack(j + 1)
				stack = stack[:len(stack)-1]
			}
		}
	}

	callBack(0)
	return ans
}

func convertIP(data string) []string {
	var ans []string
	segment := make([]string, 4)

	var callBack func(idx, si int)
	callBack = func(idx, si int) {
		if si == 4 {
			if idx == len(data) {
				ip := strings.Join(segment, ".")
				ans = append(ans, ip)
			}
			return
		}

		if idx == len(data) {
			return
		}

		if data[idx] == '0' {
			segment[si] = "0"
			callBack(idx+1, si+1)
		}

		for i := idx; i < len(data); i++ {
			num, _ := strconv.Atoi(data[idx : i+1])
			if num > 0 && num <= 255 {
				segment[si] = data[idx : i+1]
				callBack(i+1, si+1)
			} else {
				break
			}
		}
	}

	callBack(0, 0)
	return ans
}

func main() {
	fmt.Println(convertIP("0000"))
}
