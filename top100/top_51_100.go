package main

import "fmt"

func detectCycle(head *ListNode) *ListNode {
	hm := make(map[*ListNode]struct{})

	for head != nil {
		if _, ok := hm[head]; ok {
			return head
		}
		hm[head] = struct{}{}
		head = head.Next
	}

	return nil
}

func maxProduct(nums []int) int {
	maxF, minF, ans := nums[0], nums[0], nums[0]
	for i := 0; i < len(nums); i++ {
		mx, mi := maxF, minF
		maxF = max(mx*nums[i], max(nums[i], mi*nums[i]))
		minF = min(mi*nums[i], min(nums[i], mx*nums[i]))
		ans = max(maxF, ans)
	}

	return ans
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}

	pa := headA
	pb := headB

	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}

	return pa
}

func majorityElement(nums []int) int {
	var m, count int
	for _, value := range nums {
		if count == 0 {
			m = value
		}
		if m != value {
			count--
		} else {
			count++
		}
	}

	return m
}

func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}

	return pre
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	if root.Val == p.Val || root.Val == q.Val {
		return root
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}

	return left
}

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

	return lo
}

func searchMatrix(matrix [][]int, target int) bool {
	var clo []int
	for i := 0; i < len(matrix); i++ {
		clo = append(clo, matrix[i][0])
	}
	if len(clo) == 0 {
		return false
	}

	i := Binary(clo, target)
	clo = matrix[i]
	j := Binary(clo, target)
	return matrix[i][j] == target
}

func moveZeroes(nums []int) {
	left, right, n := 0, 0, len(nums)
	for right < n {
		if nums[right] != 0 {
			nums[left], nums[right] = nums[right], nums[left]
			left++
		}
		right++
	}
}

func rob(root *TreeNode) int {
	v := dfs(root)
	return max(v[0], v[1])
}

func dfs(root *TreeNode) []int {
	if root == nil {
		return []int{0, 0}
	}

	l, r := dfs(root.Left), dfs(root.Right)
	selected := root.Val + l[1] + r[1]
	notSelected := max(l[0], l[1]) + max(r[0], r[1])

	return []int{selected, notSelected}
}

func hammingDistance(x int, y int) int {
	var diff int
	for x = x ^ y; x != 0; x &= x - 1 {
		diff++
	}

	return diff
}

func diameterOfBinaryTree(root *TreeNode) int {
	dc := 1
	var deep func(*TreeNode) int
	deep = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		l := deep(node.Left)
		r := deep(node.Right)

		dc = max(dc, l+r+1)
		return max(l, r) + 1
	}
	deep(root)
	return dc - 1
}

func subarraySum(nums []int, k int) int {
	var count int
	for i := 0; i < len(nums); i++ {
		ck := k - nums[i]
		if ck == 0 {
			count++
		}
		for j := i + 1; j < len(nums); j++ {
			ck -= nums[j]
			if ck == 0 {
				count++
			}
		}
	}

	return count
}

func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	if root1 == nil {
		return root2
	}
	if root2 == nil {
		return root1
	}

	root1.Val += root2.Val
	root1.Left = mergeTrees(root1.Left, root2.Left)
	root1.Right = mergeTrees(root1.Right, root2.Right)

	return root1
}

func countSubstrings(s string) int {
	n := len(s)
	var count int
	for i := 0; i < 2*n+1; i++ {
		l, r := i/2, i/2+i%2
		for l >= 0 && r < n && s[l] == s[r] {
			l--
			r++
			count++
		}
	}

	return count
}

func dailyTemperatures(temperatures []int) []int {
	var res []int
	for i := 0; i < len(temperatures); i++ {
		dc := 0
		for j := i + 1; j < len(temperatures); j++ {
			if temperatures[j] <= temperatures[i] && j == len(temperatures)-1 {
				dc = 0
				break
			}
			dc++

			if temperatures[j] > temperatures[i] {
				break
			}

		}
		res = append(res, dc)
	}

	return res
}

func main() {
	fmt.Println(dailyTemperatures([]int{34, 80, 80, 80, 34, 80, 80, 80, 34, 34}))
}
