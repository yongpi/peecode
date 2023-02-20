package main

import "fmt"

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

func climbStairs(stairs []int) int {
	dp := make([]int, len(stairs)+1)

	for i := 2; i <= len(stairs); i++ {
		dp[i] = min(dp[i-1]+stairs[i-1], dp[i-2]+stairs[i-2])
	}

	return dp[len(stairs)]
}

func climbStairsEasy(stairs []int) int {
	var first, second int
	for i := 2; i <= len(stairs); i++ {
		third := min(second+stairs[i-1], first+stairs[i-2])
		first, second = second, third
	}

	return second
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func stealHouse(money []int) int {
	dp := make([]int, len(money))
	dp[0] = money[0]
	dp[1] = max(money[0], money[1])

	for i := 2; i < len(money); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+money[i])
	}

	return dp[len(money)-1]
}

func stealHouseEasy(money []int) int {
	first := money[0]
	second := max(money[0], money[1])

	for i := 2; i < len(money); i++ {
		first, second = second, max(second, first+money[i])
	}

	return second
}

func paintHouse(house [][3]int) int {
	n := len(house)
	dp := make([][3]int, n)
	for i := 0; i < 3; i++ {
		dp[0][i] = house[0][i]
	}

	for i := 1; i < n; i++ {
		for j := 0; j < 3; j++ {
			dp[i][j] = min(dp[i-1][(j+1)%3], dp[i-1][(j+2)%3]) + house[i][j]
		}
	}

	return min(dp[n-1][0], min(dp[n-1][1], dp[n-1][2]))
}

func paintHouseEasy(house [][3]int) int {
	n := len(house)
	dp := house[0]

	for i := 1; i < n; i++ {
		var nd [3]int
		for j := 0; j < 3; j++ {
			nd[j] = min(dp[(j+1)%3], dp[(j+2)%3]) + house[i][j]
		}
		dp = nd
	}

	return min(dp[0], min(dp[1], dp[2]))
}

func revertZero(num []int) int {
	n := len(num)
	dp := make([][2]int, n)

	if num[0] == 1 {
		dp[0][0] = 1
	}
	if num[0] == 0 {
		dp[0][1] = 1
	}

	for i := 1; i < n; i++ {
		dp[i][0] = dp[i-1][0] + num[i]
		var incr int
		if num[i] == 0 {
			incr = 1
		}
		dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + incr
	}

	return min(dp[n-1][0], dp[n-1][1])
}

func revertZeroEasy(num []int) int {
	n := len(num)
	zero := num[0]
	one := 0
	if num[0] == 0 {
		one = 1
	}

	for i := 1; i < n; i++ {
		var incr int
		if num[i] == 0 {
			incr = 1
		}

		one = min(one, zero) + incr
		zero = zero + num[i]
	}

	return min(one, zero)
}

func sameMaxSubCount(s1, s2 string) int {
	n, m := len(s1), len(s2)

	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, m+1)
	}

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if s1[i-1] == s2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}

	return dp[n][m]
}

func crossSub(s1, s2, s3 string) bool {
	n, m := len(s1), len(s2)
	dp := make([][]bool, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]bool, m+1)
	}
	dp[0][0] = true
	for i := 1; i <= n; i++ {
		dp[i][0] = dp[i-1][0] && s3[i-1] == s1[i-1]
	}

	for j := 1; j <= m; j++ {
		dp[0][j] = dp[0][j-1] && s3[j-1] == s2[j-1]
	}

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if s3[i+j-1] == s1[i-1] {
				dp[i][j] = dp[i-1][j]
			}
			if s3[i+j-1] == s2[j-1] {
				dp[i][j] = dp[i][j-1]
			}
		}
	}

	return dp[n][m]
}

func jumpSub(s1, s2 string) int {
	n, m := len(s1), len(s2)

	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, m+1)
		dp[i][0] = 1
	}

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if s1[i-1] == s2[j-1] {
				dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}

	return dp[n][m]
}

func robotMatrix(m, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}

	for i := 1; i < n; i++ {
		dp[0][i] = 1
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}

	return dp[m-1][n-1]
}

func triangleMineSum(triangle [][]int) int {
	m := len(triangle)
	dp := make([][]int, m)

	for i := 0; i < m; i++ {
		dp[i] = make([]int, i+1)
	}

	dp[0][0] = triangle[0][0]

	for i := 1; i < m; i++ {
		dp[i][0] = triangle[i][0] + dp[i-1][0]
		dp[i][i] = dp[i-1][i-1] + triangle[i][i]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < len(triangle[i])-1; j++ {
			dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j]
		}
	}

	mv := dp[m-1][0]
	for i := 1; i < len(dp[m-1]); i++ {
		mv = min(mv, dp[m-1][i])
	}

	return mv
}

func oneCoin(coins []int, target int) bool {
	n := len(coins)
	dp := make([][]bool, n)

	for i := 0; i < n; i++ {
		dp[i] = make([]bool, target+1)
	}

	dp[0][0] = true
	dp[0][coins[0]] = true

	for i := 1; i < n; i++ {
		for j := 1; j <= target; j++ {
			dp[i][j] = dp[i-1][j]
			if j >= coins[i] {
				dp[i][j] = dp[i][j] || dp[i-1][j-coins[i]]
			}
		}
	}

	return dp[n-1][target]
}

func oneCoinEasy(coins []int, target int) bool {
	n := len(coins)
	dp := make([]bool, target+1)

	dp[0] = true
	dp[coins[0]] = true

	for i := 1; i < n; i++ {
		for j := target; j >= 0; j-- {
			dp[j] = dp[j]
			if j >= coins[i] {
				dp[j] = dp[j] || dp[j-coins[i]]
			}
		}
	}

	return dp[target]
}

func repeatCoin(coins []int, target int) int {
	n := len(coins)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, target+1)
		for j := 0; j <= target; j++ {
			dp[i][j] = n
		}
	}

	dp[0][coins[0]] = 1
	for i := 1; i < n; i++ {
		for j := 1; j <= target; j++ {
			dp[i][j] = dp[i-1][j]
			if j >= coins[i] {
				dp[i][j] = min(dp[i][j], dp[i][j-coins[i]]+1)
			}
		}
	}

	return dp[n-1][target]
}

func repeatCoinEasy(coins []int, target int) int {
	n := len(coins)
	dp := make([]int, target+1)
	for i := 0; i <= target; i++ {
		dp[i] = i
	}

	dp[coins[0]] = 1

	for i := 1; i < n; i++ {
		for j := 1; j <= target; j++ {
			dp[j] = dp[j]
			if j >= coins[i] {
				dp[j] = min(dp[j], dp[j-coins[i]]+1)
			}
		}
	}

	return dp[target]
}

func sortSubListNum(num []int, target int) int {
	dp := make([]int, target+1)

	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, nv := range num {
			if i >= nv {
				dp[i] += dp[i-nv]
			}
		}
	}

	return dp[target]
}

func regular(s, p string) bool {
	m, n := len(s), len(p)

	dp := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]bool, n+1)
	}

	dp[0][0] = true

	match := func(i, j int) bool {
		if i <= 0 {
			return false
		}
		if p[j] == '.' {
			return true
		}
		if p[j] == s[i] {
			return true
		}

		return false
	}

	for i := 0; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j] || dp[i][j-2]
				if match(i-1, j-2) {
					dp[i][j] = dp[i][j] || dp[i-1][j]
				}
			} else {
				if match(i-1, j-1) {
					dp[i][j] = dp[i][j] || dp[i-1][j-1]
				}
			}
		}
	}

	return dp[m][n]

}

func rain(rain []int) int {
	var stack []int
	var ans int
	for i := 0; i < len(rain); i++ {
		for len(stack) > 0 && rain[stack[len(stack)-1]] < rain[i] {
			ci := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				break
			}

			ri := stack[len(stack)-1]
			mh := min(rain[ri], rain[i])
			width := i - ri - 1

			ans += (mh - rain[ci]) * width
		}

		stack = append(stack, i)
	}

	return ans
}

func main() {
	fmt.Println(sortSubListNum([]int{1, 2, 3}, 4))
}
