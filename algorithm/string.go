package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

func decode(num string) string {
	stack := make([]byte, 0)
	for i := 0; i < len(num); i++ {
		if num[i] != ']' {
			stack = append(stack, num[i])
		} else {
			var tmp, count []byte
			for len(stack) > 0 {
				last := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if last == '[' {
					break
				}
				tmp = append(tmp, last)

			}

			for len(stack) > 0 {
				last := stack[len(stack)-1]
				if last >= '0' && last <= '9' {
					count = append(count, last)
					stack = stack[:len(stack)-1]
				} else {
					break
				}
			}

			for i := 0; i < len(tmp)/2; i++ {
				tmp[i], tmp[len(tmp)-1-i] = tmp[len(tmp)-1-i], tmp[i]
			}

			for i := 0; i < len(count)/2; i++ {
				count[i], count[len(count)-1-i] = count[len(count)-1-i], count[i]
			}

			nc, _ := strconv.Atoi(string(count))
			for i := 0; i < nc; i++ {
				stack = append(stack, tmp...)
			}
		}
	}

	return string(stack)
}

func diffString(vs string, sub string) []int {
	hm := make(map[byte]int)
	for i := 0; i < len(sub); i++ {
		hm[sub[i]]++
	}

	ans := make([]int, 0)
	li, ri := 0, 0
	vm := make(map[byte]int)
	var count int
	for ri >= li && ri < len(vs) {
		vv := vs[ri]
		ri++
		if hm[vv] > 0 {
			vm[vv]++
			if hm[vv] == vm[vv] {
				count++
			}
		}

		for ri-li >= len(sub) {
			if count == len(hm) {
				ans = append(ans, li)
			}

			lv := vs[li]
			if hm[lv] > 0 {
				if vm[lv] == hm[lv] {
					count--
				}
				vm[lv]--
				if vm[lv] == 0 {
					delete(vm, lv)
				}
			}

			li++
		}
	}

	return ans
}

func hwStr(s string) int {
	if len(s) <= 1 {
		return 1
	}

	var isHw func(data string, i, j int) int
	isHw = func(data string, i, j int) int {
		var ct int
		for ; i >= 0 && j < len(data) && data[i] == data[j]; i, j = i-1, j+1 {
			ct++
		}

		return ct
	}

	var count int
	for i := 0; i < len(s); i++ {
		count += isHw(s, i, i)
		if i < len(s)-1 {
			count += isHw(s, i, i+1)
		}
	}

	return count
}

func notRepeatMaxSub(data string) int {
	hm := make(map[byte]int)
	li, ri, mc := 0, 0, 0
	for ri >= li && ri < len(data) {
		hm[data[ri]]++

		for hm[data[ri]] > 1 && li <= ri {
			hm[data[li]]--
			li++
		}

		mc = max(mc, ri-li+1)
		ri++
	}

	return mc
}

func containSub(data string, sub string) string {
	sm := make(map[byte]int)
	dm := make(map[byte]int)
	var ans string

	for i := 0; i < len(sub); i++ {
		sm[sub[i]]++
	}

	li, ri, valid := 0, 0, 0
	for ri >= li && ri < len(data) {
		value := data[ri]
		if sm[value] == 0 {
			ri++
			continue
		}

		dm[value]++
		if dm[value] == sm[value] {
			valid++
		}

		if valid < len(sm) {
			ri++
			continue
		}

		if ans == "" || len(ans) > ri-li+1 {
			ans = data[li : ri+1]
		}

		for ri >= li {
			lv := data[li]
			li++
			if sm[lv] > 0 {
				dm[lv]--
				if dm[lv] < sm[lv] {
					valid--
					break
				}
			}
			if ans == "" || len(ans) > ri-li+1 {
				ans = data[li : ri+1]
			}
			li++
		}
		ri++
	}

	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func subLink(data string, subs []string) bool {
	hm := make(map[string]bool)
	for _, item := range subs {
		hm[item] = true
	}

	dp := make([]bool, len(data)+1)
	dp[0] = true

	for i := 1; i <= len(data); i++ {
		for j := 0; j < i; j++ {
			if dp[j] && hm[data[j:i]] {
				dp[i] = true
				break
			}

		}
	}

	return dp[len(data)]
}

func allList(data string) []string {
	list := make([]byte, 0)
	ans := make([]string, 0)
	used := make(map[byte]bool)
	subList := make([]byte, 0)

	for i := 0; i < len(data); i++ {
		list = append(list, data[i])
	}

	var callBack func()
	callBack = func() {
		if len(subList) == len(list) {
			var tmp []byte
			tmp = append(tmp, subList...)
			ans = append(ans, string(tmp))
			return
		}

		for _, value := range list {
			if !used[value] {
				subList = append(subList, value)
				used[value] = true
				callBack()
				subList = subList[:len(subList)-1]
				used[value] = false
			}
		}
	}

	callBack()
	return ans
}

func maxProducer(words []string) int {
	hm := make(map[int]int)
	for _, word := range words {
		value := 0
		for _, b := range word {
			value |= 1 << (b - 'a')
		}
		if len(word) > hm[value] {
			hm[value] = len(word)
		}
	}

	var max = 0
	for k1, x := range hm {
		for k2, y := range hm {
			if k1&k2 == 0 && x*y > max {
				max = x * y
			}
		}
	}

	return max
}

func findMinDifference(timePoints []string) int {
	sort.Strings(timePoints)

	getInt := func(value byte) int {
		return int(value - '0')
	}

	getMinute := func(timeStr string) int {
		tv := (getInt(timeStr[0])*10+getInt(timeStr[1]))*60 + getInt(timeStr[3])*10 + getInt(timeStr[4])
		return tv
	}

	mv := math.MaxInt32

	for i := 1; i < len(timePoints); i++ {
		if mv > getMinute(timePoints[i])-getMinute(timePoints[i-1]) {
			mv = getMinute(timePoints[i]) - getMinute(timePoints[i-1])
		}
	}

	if mv > (1440 + getMinute(timePoints[0]) - getMinute(timePoints[len(timePoints)-1])) {
		mv = 1440 + getMinute(timePoints[0]) - getMinute(timePoints[len(timePoints)-1])
	}

	return mv
}

func main() {
	fmt.Println(findMinDifference([]string{"23:24", "23:18", "00:02"}))
}
