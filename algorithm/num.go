package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

func squareNum(n int) int {
	dp := make([]int, n+1)
	dp[0] = 0

	for i := 1; i <= n; i++ {
		dp[i] = i
		for j := 1; j*j <= i; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}

	return dp[n]
}

func oneBit(n int) []int {
	highBit := 0
	ans := make([]int, n+1)
	for i := 1; i <= n; i++ {
		if i&(i-1) == 0 {
			highBit = i
		}
		ans[i] = ans[i-highBit] + 1
	}

	return ans
}

type intLink struct {
	Value []int
	Next  *intLink
	Pre   *intLink
}

func InsertIndex(head *intLink, index int, value []int) *intLink {
	tl := &intLink{Value: value}
	if index == 0 {
		tl.Next = head
		head.Pre = tl

		return tl
	}

	cur := head
	for i := 0; i < index; i++ {
		next := cur.Next
		if next == nil {
			cur.Next = tl
			tl.Pre = cur
			return head
		}
		cur = next
	}

	pre := cur.Pre

	if pre != nil {
		pre.Next = tl
	}

	cur.Pre = tl
	tl.Next = cur
	tl.Pre = pre

	return head
}

func intLinkValues(head *intLink) [][]int {
	var ans [][]int
	for head != nil {
		ans = append(ans, head.Value)
		head = head.Next
	}

	return ans
}
func peopleHeight(people [][]int) [][]int {
	sort.Slice(people, func(i, j int) bool {
		return people[i][0] > people[j][0] || (people[i][0] == people[j][0] && people[i][1] < people[j][1])
	})

	head := &intLink{Value: people[0]}
	for i := 1; i < len(people); i++ {
		index := people[i][1]
		head = InsertIndex(head, index, people[i])
	}

	return intLinkValues(head)
}

func min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

func halfCount(num []int) bool {
	var sum int
	for _, value := range num {
		sum += value
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2

	dp := make([][]bool, len(num))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, target+1)
		dp[i][0] = true
	}

	dp[0][num[0]] = true

	for i := 1; i < len(dp); i++ {
		for j := 1; j <= target; j++ {
			if num[i] <= j {
				dp[i][j] = dp[i-1][j] || dp[i-1][j-num[i]]
			}
		}
	}

	return dp[len(num)-1][target]
}

func halfCountOne(num []int) bool {
	var sum int
	for _, value := range num {
		sum += value
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2

	dp := make([]bool, target+1)
	dp[0] = true

	for i := 0; i < len(num)-1; i++ {
		for j := target; j >= 1; j-- {
			if j >= num[i] {
				dp[j] = dp[j] || dp[j-num[i]]
			}
		}
	}

	return dp[target]
}

func notExistNum(num []int, n int) []int {
	for _, value := range num {
		value = (value - 1) % n
		num[value] += n
	}

	ans := make([]int, 0)
	for i := 0; i < len(num); i++ {
		if num[i] <= n {
			ans = append(ans, i+1)
		}
	}

	return ans
}

func HMDistance(x, y int) int {
	var distance int
	for x = x ^ y; x != 0; x = x&x - 1 {
		distance++
	}

	return distance
}

func threeSumZero(num []int) [][]int {
	ans := make([][]int, 0)
	sort.Ints(num)

	for i := 0; i < len(num); i++ {
		if num[i] > 0 {
			break
		}

		if i > 0 && num[i] == num[i-1] {
			continue
		}

		second := i + 1
		three := len(num) - 1

		for second < three {
			value := num[i] + num[second] + num[three]

			if value == 0 {
				ans = append(ans, []int{num[i], num[second], num[three]})
				for second < three && num[second] == num[second+1] {
					second++
				}

				for three > second && num[three] == num[three-1] {
					three--
				}

				second++
				three--
			}

			if value > 0 {
				three--
			}
			if value < 0 {
				second++
			}
		}
	}

	return ans
}

func maxSumSub(num []int) int {
	mv := math.MinInt64

	for i := 1; i < len(num); i++ {
		if num[i] < num[i]+num[i-1] {
			num[i] = num[i] + num[i-1]
		}

		mv = max(num[i], mv)
	}

	return mv
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func longSeq(num []int) int {
	hm := make(map[int]bool)
	for _, value := range num {
		hm[value] = true
	}
	var long int
	for _, item := range num {
		if hm[item-1] {
			continue
		}

		tm := 1
		for hm[item+1] {
			item++
			tm++
		}
		long = max(long, tm)
	}

	return long
}

func cutRope(rope int) float64 {
	n := rope / 3
	b := rope % 3
	if b == 0 {
		return math.Pow(3, float64(n))
	}
	if b == 1 {
		return math.Pow(3, float64(n-1)) * 4
	}

	return math.Pow(3, float64(n)) * 2
}

func quickPow(data, n int) int {
	if n == 0 {
		return 1
	}

	ans := quickPow(data, n/2)
	if n%2 == 0 {
		return ans * ans
	}

	return ans * ans * data
}

func quickPow2(data, n int) int {
	p := data
	ans := 1
	for n > 0 {
		if n%2 == 1 {
			ans = ans * p
		}
		p = p * p
		n = n / 2
	}

	return ans
}

func searchNList(n int) int {
	start, digit, count := 1, 1, 9
	for n > count {
		n -= count
		start = start * 10
		digit++
		count = 9 * start * digit
	}
	num := start + (n-1)/digit
	numStr := fmt.Sprintf("%d", num)
	index := (n - 1) % digit

	return int(numStr[index] - '0')

}

func joinNum(num []int) int {
	sl := make([]string, 0)
	for _, value := range num {
		sl = append(sl, fmt.Sprintf("%d", value))
	}

	sort.Slice(sl, func(i, j int) bool {
		x, y := sl[i], sl[j]
		return fmt.Sprintf("%s%s", x, y) < fmt.Sprintf("%s%s", y, x)
	})

	ans, _ := strconv.Atoi(strings.Join(sl, ""))
	return ans
}

func numToString(num int) int {
	var sl []byte
	ns := fmt.Sprintf("%d", num)
	for i := 0; i < len(ns); i++ {
		sl = append(sl, ns[i])
	}

	dp := make([]int, len(ns)+1)
	dp[1], dp[0] = 1, 1
	for i := 2; i <= len(ns); i++ {
		dp[i] = dp[i-1]
		tmp := ns[i-2 : i]
		if tmp >= "10" && tmp <= "25" {
			dp[i] += dp[i-2]
		}
	}

	return dp[len(sl)]
}

func numToStringEasy(num int) int {
	var sl []byte
	ns := fmt.Sprintf("%d", num)
	for i := 0; i < len(ns); i++ {
		sl = append(sl, ns[i])
	}

	first, second, third := 1, 1, 0
	for i := 1; i < len(ns); i++ {
		third = second
		tmp := ns[i-1 : i+1]
		if tmp >= "10" && tmp <= "25" {
			third += first
		}

		first, second = second, third
	}

	return third
}

func primeNum(n int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	p2, p3, p5 := 1, 1, 1
	for i := 2; i <= n; i++ {
		x2, x3, x5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
		dp[i] = min(x2, min(x3, x5))
		if dp[i] == x2 {
			p2++
		}
		if dp[i] == x3 {
			p3++
		}
		if dp[i] == x5 {
			p5++
		}
	}

	return dp[n]
}

func twoNum(num []int) (int, int) {
	sum := num[0]
	for i := 1; i < len(num); i++ {
		sum = sum ^ num[i]
	}

	n := 1
	for sum&n == 0 {
		n = n << 1
	}

	var a, b int
	for _, value := range num {
		if value&n == 0 {
			a = a ^ value
			continue
		}

		b = b ^ value
	}

	return a, b
}

func threeOneNum(num []int) int {
	var ans int32
	for i := 0; i < 31; i++ {
		var count int
		for _, value := range num {
			if value>>i&1 == 1 {
				count++
			}
		}
		if count%3 > 0 {
			ans |= 1 << i
		}
	}

	return int(ans)
}

func DiceChance(n int) []float64 {
	ans := make([]float64, 5*n+1)
	dp := make([][]float64, n+1)

	for i := 0; i < len(dp); i++ {
		dp[i] = make([]float64, 6*n+1)
	}

	chance := float64(1) / float64(6)
	for i := 1; i <= 6; i++ {
		dp[1][i] = chance
	}

	for i := 2; i <= n; i++ {
		for j := i; j <= 6*i; j++ {
			for k := 1; k <= 6; k++ {
				if j > k {
					dp[i][j] += dp[i-1][j-k] * chance
				} else {
					break
				}
			}
		}
	}

	for i := 0; i <= 5*n; i++ {
		ans[i] = dp[n][n+i]
	}

	return ans
}

func DiceChanceEasy(n int) []float64 {
	dp := make([]float64, 6)
	chance := float64(1) / float64(6)
	for i := 0; i < 6; i++ {
		dp[i] = chance
	}

	for i := 2; i <= n; i++ {
		tmp := make([]float64, 5*i+1)
		for j := 0; j < len(dp); j++ {
			for k := 0; k < 6; k++ {
				tmp[j+k] += dp[j] * chance
			}
		}
		dp = tmp
	}

	return dp
}

func add(a, b int) int {
	if b == 0 {
		return a
	}
	return add(a^b, (a&b)<<1)
}

func divide(a, b int) int {
	if b == 0 {
		return math.MinInt32
	}

	rev := false
	if a > 0 {
		a = -a
		rev = !rev
	}
	if b > 0 {
		b = -b
		rev = !rev
	}

	res := []int{b}
	// 都是负值
	for y := b; y >= a-y; {
		y += y
		res = append(res, y)
	}

	var ans int
	for i := len(res) - 1; i >= 0; i-- {
		if res[i] >= a {
			ans |= 1 << i
			a -= res[i]
		}
	}

	if rev {
		return -ans
	}

	return ans

}

func main() {
	fmt.Println(divide(-3, -9))
}
