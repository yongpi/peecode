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

func main() {
	root := &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val: 2,
			Left: &TreeNode{
				Val: 4,
			},
			Right: &TreeNode{
				Val: 5,
			},
		},
		Right: &TreeNode{
			Val: 3,
			Left: &TreeNode{
				Val: 6,
			},
			Right: &TreeNode{
				Val: 7,
			},
		},
	}
	fmt.Println(invertTree(root))
}
