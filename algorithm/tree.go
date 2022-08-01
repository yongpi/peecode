package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func pre(root *TreeNode) []int {
	var res []int
	var pred func(root *TreeNode)

	pred = func(root *TreeNode) {
		if root == nil {
			return
		}
		res = append(res, root.Val)
		pred(root.Left)
		pred(root.Right)
	}

	pred(root)
	return res
}

func pre2(root *TreeNode) []int {
	var res []int
	stack := make([]*TreeNode, 0)
	cur := root

	for cur != nil || len(stack) > 0 {
		if cur != nil {
			res = append(res, cur.Val)
			stack = append(stack, cur)
			cur = cur.Left
		}
		if cur == nil {
			cur = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cur = cur.Right
		}
	}

	return res
}

func mid(root *TreeNode) []int {
	var res []int
	var midd func(root *TreeNode)

	midd = func(root *TreeNode) {
		if root == nil {
			return
		}
		midd(root.Left)
		res = append(res, root.Val)
		midd(root.Right)
	}

	midd(root)
	return res
}

func mid2(root *TreeNode) []int {
	var res []int
	stack := make([]*TreeNode, 0)
	cur := root

	for cur != nil || len(stack) > 0 {
		if cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		if cur == nil {
			cur = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res = append(res, cur.Val)
			cur = cur.Right
		}
	}

	return res
}

func after(root *TreeNode) []int {
	var res []int
	var aftd func(root *TreeNode)

	aftd = func(root *TreeNode) {
		if root == nil {
			return
		}
		aftd(root.Left)
		aftd(root.Right)
		res = append(res, root.Val)
	}

	aftd(root)
	return res
}

func after2(root *TreeNode) []int {
	var res []int
	stack := make([]*TreeNode, 0)
	cur := root
	var pre *TreeNode

	for cur != nil || len(stack) > 0 {
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}

		cur = stack[len(stack)-1]
		if cur.Right == nil || cur.Right == pre {
			stack = stack[:len(stack)-1]
			res = append(res, cur.Val)
			pre = cur
			cur = nil
		} else {
			cur = cur.Right
		}
	}

	return res
}

func level(root *TreeNode) []int {
	var res []int
	stack := make([]*TreeNode, 0)
	stack = append(stack, root)

	for len(stack) > 0 {
		size := len(stack)
		for i := 0; i < size; i++ {
			res = append(res, stack[i].Val)
			if stack[i].Left != nil {
				stack = append(stack, stack[i].Left)
			}
			if stack[i].Right != nil {
				stack = append(stack, stack[i].Right)
			}
		}

		stack = stack[size:]

	}

	return res
}

func invertTree(root *TreeNode) *TreeNode {
	invert(root)
	return root
}

func invert(root *TreeNode) {
	if root == nil {
		return
	}

	left := root.Left
	right := root.Right

	root.Right = left
	root.Left = right

	invert(left)
	invert(right)
}

func invertTree2(root *TreeNode) *TreeNode {
	stack := make([]*TreeNode, 0)
	stack = append(stack, root)

	for len(stack) > 0 {
		size := len(stack)
		for i := 0; i < size; i++ {
			node := stack[i]
			left := node.Left
			right := node.Right

			node.Left = node.Right
			node.Right = left

			if left != nil {
				stack = append(stack, left)
			}
			if right != nil {
				stack = append(stack, right)
			}
		}

		stack = stack[size:]
	}

	return root
}

func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	stack := make([]*TreeNode, 0)
	cur := root
	var head, tail *TreeNode
	for cur != nil || len(stack) > 0 {
		if cur != nil {
			if head == nil {
				head = &TreeNode{Val: cur.Val}
				tail = head
			} else {
				tail.Right = &TreeNode{Val: cur.Val}
				tail = tail.Right
			}
			stack = append(stack, cur)
			cur = cur.Left
		}
		if cur == nil {
			cur = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cur = cur.Right
		}
	}

	root = head
}

func lastParent(root, left, right *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	if root.Val == left.Val || root.Val == right.Val {
		return root
	}

	lt := lastParent(root.Left, left, right)
	rt := lastParent(root.Right, left, right)

	if lt != nil && rt != nil {
		return root
	}

	if lt == nil {
		return rt
	}

	return lt
}

func lastParentNormal(root, left, right *TreeNode) *TreeNode {
	parent := make(map[int]*TreeNode)
	visited := make(map[int]bool)

	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Left != nil {
			parent[node.Left.Val] = node
			dfs(node.Left)
		}
		if node.Right != nil {
			parent[node.Right.Val] = node
			dfs(node.Right)
		}
	}

	dfs(root)

	for left != nil {
		visited[left.Val] = true
		left = parent[left.Val]
	}

	for right != nil {
		if visited[right.Val] {
			return right
		}

		right = parent[right.Val]
	}

	return nil
}

func robTree(root *TreeNode) int {
	var dp func(node *TreeNode) []int
	dp = func(node *TreeNode) []int {
		if node == nil {
			return []int{0, 0}
		}
		lv := dp(node.Left)
		rv := dp(node.Right)
		selected := node.Val + lv[1] + rv[1]
		notSelected := max(lv[0], lv[1]) + max(rv[0], rv[1])

		return []int{selected, notSelected}
	}

	ans := dp(root)
	return max(ans[0], ans[1])
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sumTree(root *TreeNode, target int) int {
	var dp func(root *TreeNode, target int) int
	dp = func(root *TreeNode, target int) int {
		var res int
		if root == nil {
			return res
		}
		if root.Val == target {
			res++
		}
		res += dp(root.Left, target-root.Val)
		res += dp(root.Right, target-root.Val)

		return res
	}

	res := dp(root, target)
	res += sumTree(root.Left, target)
	res += sumTree(root.Right, target)
	return res
}

func search2Sum(root *TreeNode) *TreeNode {
	var sum int
	var df func(node *TreeNode)
	df = func(node *TreeNode) {
		if node == nil {
			return
		}
		df(node.Right)
		sum += node.Val
		node.Val = sum
		df(node.Left)

	}

	df(root)
	return root
}

func longDistance(root *TreeNode) int {
	if root == nil {
		return 0
	}
	var md int
	var df func(node *TreeNode) int
	df = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		lv := df(node.Left)
		rv := df(node.Right)
		md = max(md, lv+rv+1)
		return max(lv, rv) + 1
	}

	df(root)
	return md - 1
}

func mergeTree(root1, root2 *TreeNode) *TreeNode {
	if root1 == nil {
		return root2
	}
	if root2 == nil {
		return root1
	}

	root1.Val += root2.Val
	root1.Left = mergeTree(root1.Left, root2.Left)
	root1.Right = mergeTree(root1.Right, root2.Right)

	return root1
}

func buildTree(pre, mid []int) *TreeNode {
	root := &TreeNode{Val: pre[0]}
	stack := make([]*TreeNode, 0)
	stack = append(stack, root)

	var mi int
	for i := 1; i < len(pre); i++ {
		node := stack[len(stack)-1]
		if node.Val != mid[mi] {
			node.Left = &TreeNode{Val: pre[i]}
			stack = append(stack, node.Left)
			continue
		}

		for len(stack) > 0 && stack[len(stack)-1].Val == mid[mi] {
			node = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			mi++
		}

		node.Right = &TreeNode{Val: pre[i]}
		stack = append(stack, node.Right)
	}
	return root
}

func buildTree2(pre, mid []int) *TreeNode {
	if len(pre) == 0 {
		return nil
	}
	root := &TreeNode{Val: pre[0]}
	var mi int
	for key, value := range mid {
		if value == pre[0] {
			mi = key
		}
	}

	root.Left = buildTree2(pre[1:mi+1], mid[:mi])
	root.Right = buildTree2(pre[mi+1:], mid[mi+1:])

	return root
}

func containChild(parent, child *TreeNode) bool {
	return (parent != nil && child != nil) && isContain(parent, child) || containChild(parent.Left, child) || containChild(parent.Right, child)
}

func isContain(parent, child *TreeNode) bool {
	if child == nil {
		return true
	}

	if parent == nil || parent.Val != child.Val {
		return false
	}

	return isContain(parent.Left, child.Left) && isContain(parent.Right, child.Right)
}

func isBalance(root *TreeNode) bool {
	var deep func(node *TreeNode) int
	deep = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		ld := deep(node.Left)
		rd := deep(node.Right)

		if ld == -1 || rd == -1 || ld-rd > 1 || rd-ld > 1 {
			return -1
		}

		return max(ld, rd) + 1
	}

	return deep(root) != -1
}

func isBalance2(root *TreeNode) bool {
	var height func(node *TreeNode) int
	height = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		return max(height(node.Left), height(node.Right)) + 1
	}

	lh := height(root.Left)
	rh := height(root.Right)

	if lh-rh > 1 || rh-lh > 1 {
		return false
	}

	return isBalance2(root.Left) && isBalance2(root.Right)
}

func cutTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	root.Left = cutTree(root.Left)
	root.Right = cutTree(root.Right)

	if root.Left == nil && root.Right == nil && root.Val == 0 {
		return nil
	}

	return root
}

func main() {
	//root := &TreeNode{
	//	Val: 1,
	//	Left: &TreeNode{
	//		Val: 2,
	//		Left: &TreeNode{
	//			Val: 3,
	//		},
	//		Right: &TreeNode{
	//			Val: 4,
	//		},
	//	},
	//	Right: &TreeNode{
	//		Val: 5,
	//		//Left: &TreeNode{
	//		//	Val: 6,
	//		//},
	//		Right: &TreeNode{
	//			Val: 6,
	//		},
	//	},
	//}
	//flatten(root)
	tree := buildTree2([]int{3, 9, 20, 15, 7}, []int{9, 3, 15, 20, 7})
	fmt.Println(tree)
}
