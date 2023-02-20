package main

import "fmt"

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxIsland(matrix [][]int) int {
	n, m := len(matrix), len(matrix[0])

	var ans int
	list := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}

	var search func(i, j int) int
	search = func(i, j int) int {
		if i < 0 || i >= n || j < 0 || j >= m {
			return 0
		}

		if matrix[i][j] == 0 {
			return 0
		}

		matrix[i][j] = 0
		size := 1
		for _, item := range list {
			size += search(i+item[0], j+item[1])
		}

		return size
	}

	for i := range matrix {
		for j := range matrix[i] {
			ans = max(search(i, j), ans)
		}
	}

	return ans
}

func updateMatrix(mat [][]int) [][]int {
	n, m := len(mat), len(mat[0])
	stack := make([][]int, 0)
	ans := make([][]int, n)
	visit := make([][]bool, n)

	for i := 0; i < n; i++ {
		ans[i] = make([]int, m)
		visit[i] = make([]bool, m)
		for j := 0; j < m; j++ {
			if mat[i][j] == 0 {
				stack = append(stack, []int{i, j})
				visit[i][j] = true
			}
		}
	}

	list := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for len(stack) > 0 {
		item := stack[0]
		stack = stack[1:]
		i, j := item[0], item[1]

		for _, data := range list {
			ri, rj := i+data[0], j+data[1]
			if ri < 0 || ri >= n || rj < 0 || rj >= m || visit[ri][rj] {
				continue
			}

			ans[ri][rj] = ans[i][j] + 1
			stack = append(stack, []int{ri, rj})
			visit[ri][rj] = true
		}
	}

	return ans
}

func openLock(deadends []string, target string) int {
	start := "0000"

	if target == start {
		return 0
	}

	dead := map[string]bool{}
	for _, s := range deadends {
		dead[s] = true
	}
	if dead[start] {
		return -1
	}

	nextKey := func(key string) []string {
		var ret []string
		s := []byte(key)
		for i, b := range s {
			s[i] = b - 1
			if s[i] < '0' {
				s[i] = '9'
			}
			ret = append(ret, string(s))
			s[i] = b + 1
			if s[i] > '9' {
				s[i] = '0'
			}
			ret = append(ret, string(s))
			s[i] = b
		}

		return ret
	}

	type param struct {
		Key   string
		Count int
	}

	stack := []param{{Key: start, Count: 0}}
	hm := map[string]bool{start: true}

	for len(stack) > 0 {
		sp := stack[0]
		stack = stack[1:]

		keys := nextKey(sp.Key)
		count := sp.Count + 1
		for _, key := range keys {
			if hm[key] {
				continue
			}
			if dead[sp.Key] {
				continue
			}

			if key == target {
				return count
			}

			stack = append(stack, param{Key: key, Count: count})
			hm[key] = true
		}
	}

	return -1
}

func main() {
	fmt.Println(openLock([]string{"0201", "0101", "0102", "1212", "2002"}, "0202"))
}
