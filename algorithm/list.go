package main

import (
	"fmt"
)

func MaxAbs(num []int) int {
	maxList := make([]int, len(num))
	minList := make([]int, len(num))

	maxList[0], minList[0] = num[0], num[0]

	for i := 1; i < len(num); i++ {
		maxList[i] = max(maxList[i-1]*num[i], max(num[i], minList[i-1]*num[i]))
		minList[i] = min(minList[i-1]*num[i], min(num[i], maxList[i-1]*num[i]))
	}
	ant := maxList[0]
	for i := 1; i < len(maxList); i++ {
		ant = max(ant, maxList[i])
	}

	return ant
}

func MaxAbs2(num []int) int {
	maxV, minV, ant := num[0], num[0], num[0]

	for i := 1; i < len(num); i++ {
		maxT, minT := maxV, minV
		maxV = max(maxT*num[i], max(num[i], minT*num[i]))
		minV = min(minT*num[i], min(num[i], maxT*num[i]))

		ant = max(maxV, ant)
	}

	return ant
}

func manyNum(num []int) int {
	var count, many int
	for i := 0; i < len(num); i++ {
		if count == 0 {
			many = num[i]
		}

		if many == num[i] {
			count++
		} else {
			count--
		}
	}

	return many
}

func stealMoney(house []int) int {
	dp := make([]int, len(house))
	dp[0] = house[0]
	dp[1] = max(house[0], house[1])

	for i := 2; i < len(house); i++ {
		dp[i] = max(dp[i-2]+house[i], dp[i-1])
	}

	return dp[len(house)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

func landNum(matrix [][]int) int {
	im := len(matrix)
	jm := len(matrix[0])

	var dfs func(matrix [][]int, i, j int)
	dfs = func(matrix [][]int, i, j int) {
		// 边界条件
		if i >= im || j >= jm || i < 0 || j < 0 {
			return
		}
		if matrix[i][j] == 0 || matrix[i][j] == 2 {
			return
		}
		matrix[i][j] = 2
		dfs(matrix, i, j-1)
		dfs(matrix, i, j+1)
		dfs(matrix, i+1, j)
		dfs(matrix, i-1, j)
	}

	var count int
	for i := 0; i < im; i++ {
		for j := 0; j < jm; j++ {
			if matrix[i][j] == 1 {
				count++
				dfs(matrix, i, j)
			}
		}
	}

	return count
}

func classList(schedule [][]int, n int) bool {
	inList := make([]int, n)
	outList := make([][]int, n)

	for i := 0; i < len(schedule); i++ {
		out := outList[schedule[i][1]]
		out = append(out, schedule[i][0])
		outList[schedule[i][1]] = out

		inList[schedule[i][0]]++
	}

	var idle []int
	for i := 0; i < n; i++ {
		if inList[i] == 0 {
			idle = append(idle, i)
		}
	}

	var result []int
	for len(idle) > 0 {
		class := idle[0]
		idle = idle[1:]
		result = append(result, class)

		for _, item := range outList[class] {
			inList[item]--
			if inList[item] == 0 {
				idle = append(idle, item)
			}
		}
	}

	return len(result) == n
}

func maxSquare(matrix [][]int) int {
	maxSlide := 0
	dp := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == 1 {
				dp[i][j] = 1
				maxSlide = 1
			}
		}
	}

	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[i]); j++ {
			if dp[i][j] == 1 {
				dp[i][j] = min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1])) + 1
				maxSlide = max(dp[i][j], maxSlide)
			}
		}
	}

	return maxSlide * maxSlide
}

func otherAbs(num []int) []int {
	la := make([]int, len(num))
	ra := make([]int, len(num))

	la[0] = 1
	for i := 1; i < len(num); i++ {
		la[i] = num[i-1] * la[i-1]
	}

	ra[len(num)-1] = 1
	for i := len(num) - 2; i >= 0; i-- {
		ra[i] = ra[i+1] * num[i+1]
	}

	res := make([]int, 0)
	for i := 0; i < len(num); i++ {
		res = append(res, la[i]*ra[i])
	}

	return res
}

func otherAbsCheaper(num []int) []int {
	la := make([]int, len(num))

	la[0] = 1
	for i := 1; i < len(num); i++ {
		la[i] = num[i-1] * la[i-1]
	}

	res := make([]int, len(num))
	rs := 1
	for i := len(num) - 1; i >= 0; i-- {
		res[i] = rs * la[i]
		rs = rs * num[i]
	}

	return res
}

func slideWindow(num []int, k int) []int {
	stack := make([]int, 0)
	res := make([]int, 0)
	lp, rp := 0, 0

	for rp < len(num) {
		for len(stack) != 0 && stack[len(stack)-1] < num[rp] {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, rp)

		if stack[0] < lp {
			stack = stack[1:]
		}
		if rp-lp >= k-1 {
			res = append(res, num[stack[0]])
			lp++
		}
		rp++
	}

	return res
}

func searchMatrix(matrix [][]int, target int) bool {
	width := len(matrix[0])
	height := len(matrix)

	x, y := 0, height-1
	for x < width && y >= 0 {
		if matrix[x][y] > target {
			y--
		}
		if matrix[x][y] < target {
			x++
		}

		if matrix[x][y] == target {
			return true
		}
	}

	return false
}

func moveZero(num []int) []int {
	lp, rp := 0, 0
	for rp < len(num) {
		if num[rp] != 0 {
			num[rp], num[lp] = num[lp], num[rp]
			lp++
		}
		rp++
	}

	return num
}

func repeatNum(num []int) int {
	slow := num[0]
	fast := num[num[0]]
	for slow != fast {
		slow = num[slow]
		fast = num[num[fast]]
	}

	head := 0
	for head != slow {
		head = num[head]
		slow = num[slow]
	}

	return slow
}

func maxAesList(num []int) int {
	dp := make([]int, len(num))
	dp[0] = 1
	maxLen := 1
	for i := 1; i < len(num); i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if num[i] > num[j] {
				dp[i] = max(dp[i], dp[j]+1)
				maxLen = max(dp[i], maxLen)
			}
		}
	}

	return maxLen
}

func maxAesListCheap(num []int) int {
	stack := make([]int, 0)
	for i := 0; i < len(num); i++ {
		if len(stack) == 0 || stack[len(stack)-1] < num[i] {
			stack = append(stack, num[i])
			continue
		}

		l, r := 0, len(stack)-1
		mid := l + (r-l)/2
		for l <= r {
			if stack[mid] == num[i] {
				break
			}
			if stack[mid] > num[i] {
				r = mid - 1
			} else {
				l = mid + 1
			}
			mid = l + (r-l)/2
		}

		if mid >= len(stack)-1 {
			stack[len(stack)-1] = num[i]
		} else {
			stack[mid+1] = num[i]
		}
	}

	return len(stack)
}

func stockPrice(prices []int) int {
	n := len(prices)
	dp := make([][3]int, n)
	dp[0][0] = -prices[0]

	for i := 1; i < n; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i])
		dp[i][1] = dp[i-1][0] + prices[i]
		dp[i][2] = max(dp[i-1][2], dp[i-1][1])
	}

	return max(dp[n-1][1], dp[n-1][2])
}

func coinMoney(coins []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 0
	for i := 1; i <= target; i++ {
		dp[i] = i
		for j := 0; j < len(coins); j++ {
			if i >= coins[j] {
				dp[i] = min(dp[i], dp[i-coins[j]]+1)
			}
		}
	}

	return dp[target]
}

func calNum(eq [][]string, values []float64, qu [][]string) []float64 {
	relations := make(map[string]map[string]float64)
	relations2 := make(map[string]map[string]float64)
	ans := make([]float64, 0)
	for i, item := range eq {
		m, ok := relations[item[0]]
		if !ok {
			m = make(map[string]float64)
		}
		m[item[1]] = values[i]
		relations[item[0]] = m

		m1, ok := relations2[item[1]]
		if !ok {
			m1 = make(map[string]float64)
		}
		m1[item[0]] = 1 / values[i]
		relations2[item[1]] = m1

	}

	for _, item := range qu {
		a1 := calAbsNum(relations, item)
		if a1 == -1 {
			a1 = calAbsNum(relations2, item)
		}

		ans = append(ans, a1)
	}

	return ans
}

type stackValue struct {
	Base  string
	Value float64
}

func calAbsNum(relations map[string]map[string]float64, item []string) float64 {
	stack := make([]*stackValue, 0)
	stack = append(stack, &stackValue{Base: item[0], Value: float64(1)})

	for len(stack) > 0 {
		size := len(stack)
		for i := 0; i < size; i++ {
			rk := stack[i]
			m, ok := relations[rk.Base]
			if !ok {
				continue
			}
			rv, ok := m[item[1]]
			if ok {
				return rk.Value * rv
			}

			for key, value := range m {
				stack = append(stack, &stackValue{Base: key, Value: rk.Value * value})
			}
		}

		stack = stack[size:]
	}

	return -1
}

func targetSum(num []int, target int) int {
	var sum int
	for _, value := range num {
		sum += value
	}

	rt := sum - target
	if rt%2 != 0 {
		return 0
	}

	rt = rt / 2
	dp := make([][]int, len(num))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, rt+1)
	}

	dp[0][0] = 1

	for i := 1; i < len(num); i++ {
		for j := 0; j <= rt; j++ {
			dp[i][j] = dp[i-1][j]
			if j >= num[i] {
				dp[i][j] += dp[i-1][j-num[i]]
			}
		}
	}

	return dp[len(num)-1][rt]
}

func preSumK(num []int, k int) int {
	m := make(map[int]int)
	m[0] = 1
	var pre, count int
	for i := 0; i < len(num); i++ {
		pre += num[i]
		if v, ok := m[pre-k]; ok {
			count += v
		}
		m[pre]++
	}

	return count
}

func shortList(num []int) int {
	li, ri := 0, len(num)-1

	for ri >= li {
		if li < len(num)-2 && num[li] <= num[li+1] {
			li++
			continue
		}
		if ri > 0 && num[ri] >= num[ri-1] {
			ri--
			continue
		}
		break
	}

	return ri - li + 1
}

func cpuTask(tasks []string, n int) int {
	hm := make(map[string]int)
	var mc, same int
	for _, task := range tasks {
		hm[task]++
		if hm[task] > mc {
			mc = hm[task]
		}
	}

	for _, value := range hm {
		if value == mc {
			same++
		}
	}

	small := (mc-1)*(n+1) + same
	if small > len(tasks) {
		return small
	}

	return len(tasks)
}

func temperature(num []int) []int {
	if len(num) <= 1 {
		return []int{0}
	}

	ans := make([]int, len(num))
	stack := make([]int, 0)
	stack = append(stack, 0)

	for i := 1; i < len(num); i++ {
		for len(stack) > 0 {
			cur := stack[len(stack)-1]
			if num[i] > num[cur] {
				ans[cur] = i - cur
				stack = stack[:len(stack)-1]
			} else {
				break
			}
		}

		stack = append(stack, i)
	}
	return ans
}

func matrixSearchString(matrix [][]byte, sub string) bool {
	n, m := len(matrix), len(matrix[0])

	var check func(i, j, si int, used [][]bool) bool
	check = func(i, j, si int, used [][]bool) bool {
		if i >= n || j >= m || i < 0 || j < 0 {
			return false
		}

		if used[i][j] {
			return false
		}

		if matrix[i][j] == sub[si] {
			return true
		}

		return false
	}

	var search func(i, j int) bool
	search = func(i, j int) bool {
		used := make([][]bool, n)
		for ui, _ := range used {
			used[ui] = make([]bool, m)
		}

		if !check(i, j, 0, used) {
			return false
		}

		for si := 1; si < len(sub); si++ {
			if check(i+1, j, si, used) {
				i++
				continue
			}
			if check(i, j+1, si, used) {
				j++
				continue
			}
			if check(i-1, j, si, used) {
				i--
				continue
			}
			if check(i, j-1, si, used) {
				j--
				continue
			}
			return false
		}

		return true
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if search(i, j) {
				return true
			}
		}
	}

	return false
}

func minMatrixSum(matrix [][]int) int {
	m, n := len(matrix), len(matrix[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = matrix[0][0]
	for i := 1; i < n; i++ {
		dp[0][i] = dp[0][i-1] + matrix[0][i]
	}

	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0] + matrix[i][0]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + matrix[i][j]
		}
	}

	return dp[m-1][n-1]

}

func minMatrixSumEasy(matrix [][]int) int {
	m, n := len(matrix), len(matrix[0])
	for i := 1; i < n; i++ {
		matrix[0][i] = matrix[0][i-1] + matrix[0][i]
	}

	for i := 1; i < m; i++ {
		matrix[i][0] = matrix[i-1][0] + matrix[i][0]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1]) + matrix[i][j]
		}
	}

	return matrix[m-1][n-1]
}

func zeroTwoList(num []int) []int {
	li, ri := 0, 0

	for ri < len(num) {
		if num[ri] != 0 {
			num[ri], num[li] = num[li], num[ri]
			li++
		}
		ri++
	}

	return num
}

func threeColorSort(num []int) []int {
	two, zero := len(num)-1, 0
	for i := 0; i <= two; i++ {
		for ; num[i] == 2 && two >= i; two-- {
			num[i], num[two] = num[two], num[i]

		}

		if num[i] == 0 {
			num[i], num[zero] = num[zero], num[i]
			zero++
		}
	}

	return num
}

func allSubList(num []int) [][]int {
	stack := make([]int, 0)
	ans := make([][]int, 0)

	var callBack func(index int)
	callBack = func(index int) {
		var tmp []int
		tmp = append(tmp, stack...)
		ans = append(ans, tmp)

		for i := index; i < len(num); i++ {
			stack = append(stack, num[i])
			callBack(i + 1)
			stack = stack[:len(stack)-1]
		}
	}

	callBack(0)
	return ans
}

func mulMin(nums []int, k int) int {
	left, right, sum := 0, 0, 1
	var total int

	for right < len(nums) {
		sum *= nums[right]
		for left <= right && sum >= k {
			sum /= nums[left]
			left++
		}

		total += (right - left) + 1
		right++
	}

	return total

}

func main() {
	fmt.Println(mulMin([]int{10, 5, 2, 6}, 100))
	//fmt.Println(calNum([][]string{{"a", "b"}, {"b", "c"}, {"bc", "cd"}}, []float64{1.5, 2.5, 5.0}, [][]string{{"a", "c"}, {"c", "b"}, {"bc", "cd"}, {"cd", "bc"}}))
	//fmt.Println(matrixSearchString([][]byte{{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}}, "ABCCED"))
}
