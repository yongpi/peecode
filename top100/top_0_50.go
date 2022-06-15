package main

//import "github.com/go/src/math"

func twoSum(nums []int, target int) []int {
	hm := make(map[int]int, len(nums))

	for key, value := range nums {
		if k2, ok := hm[target-value]; ok {
			return []int{key, k2}
		}
		hm[value] = key
	}

	return nil
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var header, tail *ListNode
	carry := 0

	for l1 != nil || l2 != nil {
		var lv, rv int
		if l1 != nil {
			lv = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			rv = l2.Val
			l2 = l2.Next
		}

		hv := lv + rv + carry
		hv, carry = hv%10, hv/10
		if header == nil {
			header = &ListNode{Val: hv}
			tail = header
		} else {
			tail.Next = &ListNode{Val: hv}
			tail = tail.Next
		}
	}

	if carry > 0 {
		tail.Next = &ListNode{Val: carry}
	}

	return header
}

func lengthOfLongestSubstring(s string) int {
	hm := make(map[byte]bool, len(s))
	ri, max, n := -1, 0, len(s)

	for i := 0; i < n; i++ {
		if i > 0 {
			delete(hm, s[i-1])
		}
		for ; ri+1 < n && !hm[s[ri+1]]; ri++ {
			hm[s[ri+1]] = true
		}

		if max < ri-i+1 {
			max = ri - i + 1
		}
	}

	return max
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func getKthElement(nums1, nums2 []int, k int) int {
	var i1, i2 int

	for {
		if i1 == len(nums1) {
			return nums2[i2+k-1]
		}
		if i2 == len(nums2) {
			return nums1[i1+k-1]
		}
		if k == 1 {
			return min(nums1[i1], nums2[i2])
		}

		half := k / 2
		i1m := min(i1+half, len(nums1)) - 1
		i2m := min(i2+half, len(nums2)) - 1

		v1, v2 := nums1[i1m], nums2[i2m]
		if v1 <= v2 {
			k = k - (i1m - i1 + 1)
			i1 = i1m + 1
		} else {
			k = k - (i2m - i2 + 1)
			i2 = i2m + 2
		}
	}
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	totalLength := len(nums1) + len(nums2)
	if totalLength%2 == 1 {
		midIndex := totalLength / 2
		return float64(getKthElement(nums1, nums2, midIndex+1))
	} else {
		midIndex1, midIndex2 := totalLength/2-1, totalLength/2
		return float64(getKthElement(nums1, nums2, midIndex1+1)+getKthElement(nums1, nums2, midIndex2+1)) / 2.0
	}
	return 0
}

func longestPalindrome(s string) string {
	if s == "" {
		return ""
	}

	start, end := 0, 0
	for i := 0; i < len(s); i++ {
		l1, r1 := expandAroundCenter(s, i, i)
		l2, r2 := expandAroundCenter(s, i, i+1)

		if r1-l1 > end-start {
			start, end = l1, r1
		}

		if r2-l2 > end-start {
			start, end = l2, r2
		}
	}

	return s[start : end+1]
}

func expandAroundCenter(s string, left, right int) (int, int) {
	for ; left >= 0 && right < len(s) && s[left] == s[right]; left, right = left-1, right+1 {
	}
	return left + 1, right - 1
}

var phoneMap map[string]string = map[string]string{
	"2": "abc",
	"3": "def",
	"4": "ghi",
	"5": "jkl",
	"6": "mno",
	"7": "pqrs",
	"8": "tuv",
	"9": "wxyz",
}

var combinations []string

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}

	combinations = []string{}
	trackBack(digits, 0, "")
	return combinations
}

func trackBack(digits string, index int, combination string) {
	if index == len(digits) {
		combinations = append(combinations, combination)
		return
	}

	digit := string(digits[index])
	letter := phoneMap[digit]
	for i := 0; i < len(letter); i++ {
		trackBack(digits, index+1, combination+string(letter[i]))
	}
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Val: 0, Next: head}
	first, second := head, dummy

	for i := 0; i < n; i++ {
		first = first.Next
	}

	for ; first != nil; first = first.Next {
		second = second.Next
	}

	second.Next = second.Next.Next
	return dummy.Next
}

func isValid(s string) bool {
	if len(s)%2 != 0 {
		return false
	}

	pair := map[byte]byte{
		')': '(',
		'}': '{',
		']': '[',
	}
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if pair[s[i]] != 0 {
			if len(stack) == 0 || stack[len(stack)-1] != pair[s[i]] {
				return false
			}
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, s[i])
		}
	}

	return len(stack) == 0
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}
	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list2.Next, list1)
		return list2
	}
}

func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	return merge(lists, 0, len(lists)-1)
}

func merge(list []*ListNode, left, right int) *ListNode {
	if left >= right {
		return list[left]
	}

	mid := left + (right-left)/2
	l1 := merge(list, left, mid)
	l2 := merge(list, mid+1, right)
	return mergeTwoLists(l1, l2)
}

var result [][]int

func permute(nums []int) [][]int {
	result = make([][]int, 0)
	values := make([]int, 0)
	used := make([]bool, len(nums))
	permuteBackTrace(nums, values, used)
	return result
}

func permuteBackTrace(nums, values []int, used []bool) {
	if len(values) == len(nums) {
		result = append(result, append([]int(nil), values...))
		return
	}

	for i := 0; i < len(nums); i++ {
		if !used[i] {
			values = append(values, nums[i])
			used[i] = true
			permuteBackTrace(nums, values, used)
			used[i] = false
			values = values[:len(values)-1]
		}
	}
}

var combinationResult [][]int

func combinationSum(candidates []int, target int) [][]int {
	combinationResult = make([][]int, 0)
	values := make([]int, 0)
	combinationBackTrack(candidates, values, target, 0)
	return combinationResult

}

func combinationBackTrack(candidates, values []int, target, begin int) {
	if target <= 0 {
		if target == 0 {
			combinationResult = append(combinationResult, append([]int(nil), values...))
		}
		return
	}

	for i := begin; i < len(candidates); i++ {
		values = append(values, candidates[i])
		target -= candidates[i]
		combinationBackTrack(candidates, values, target, i)
		values = values[:len(values)-1]
		target += candidates[i]
	}
}

func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] < nums[i]+nums[i-1] {
			nums[i] = nums[i] + nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}

	return max
}

func canJump(nums []int) bool {
	move := 0
	n := len(nums)

	for i := 0; i < n; i++ {
		if i <= move {
			if move < i+nums[i] {
				move = i + nums[i]
			}
			if move >= n-1 {
				return true
			}
		} else {
			return false
		}
	}
	return false
}

func uniquePaths(m int, n int) int {
	all := make([][]int, m)
	for i := 0; i < m; i++ {
		all[i] = make([]int, n)
		all[i][0] = 1
	}

	for i := 0; i < n; i++ {
		all[0][i] = 1
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			all[i][j] = all[i-1][j] + all[i][j-1]
		}
	}

	return all[m-1][n-1]
}

func minPathSum(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	m, n := len(grid), len(grid[0])
	all := make([][]int, m)
	for i := 0; i < m; i++ {
		all[i] = make([]int, n)
	}
	all[0][0] = grid[0][0]

	for i := 1; i < m; i++ {
		all[i][0] = all[i-1][0] + grid[i][0]
	}

	for i := 1; i < n; i++ {
		all[0][i] = all[0][i-1] + grid[0][i]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			all[i][j] = min(all[i-1][j], all[i][j-1]) + grid[i][j]
		}
	}

	return all[m-1][n-1]
}

func climbStairs(n int) int {
	dp := make([]int, n+1)

	dp[0] = 1
	dp[1] = 1

	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}

	return dp[n]
}

func sortColors(nums []int) {
	sortPosition(nums, 0, len(nums)-1)
}

func sortPosition(nums []int, lo, hi int) {
	if lo >= hi {
		return
	}
	position := position(nums, lo, hi)
	sortPosition(nums, lo, position-1)
	sortPosition(nums, position+1, hi)
}

func exchange(list []int, lo, hi int) {
	lv := list[lo]
	list[lo] = list[hi]
	list[hi] = lv
}

func position(nums []int, lo, hi int) int {
	src := nums[lo]
	i := lo + 1
	j := hi

	for {
		for ; nums[i] <= src; i++ {
			if i >= hi {
				break
			}
		}
		for ; nums[j] > src; j-- {
			if j <= lo {
				break
			}
		}

		if i >= j {
			break
		}
		exchange(nums, i, j)
	}
	exchange(nums, lo, j)
	return j
}

func subsets(nums []int) [][]int {
	result := make([][]int, 0)
	mask := 1 << len(nums)

	for i := 0; i < mask; i++ {
		set := make([]int, 0)
		for key, value := range nums {
			if i>>key&1 > 0 {
				set = append(set, value)
			}
		}
		result = append(result, set)
	}

	return result
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isValidBST(root *TreeNode) bool {
	stack := make([]*TreeNode, 0)
	cur := root
	var res []int

	for cur != nil || len(stack) > 0 {
		if cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		if cur == nil {
			cur = stack[len(stack)-1]
			stack = stack[:len(stack)-1]

			if len(res) == 0 {
				res = append(res, cur.Val)
			} else if res[0] >= cur.Val {
				return false
			} else {
				res[0] = cur.Val
			}
			cur = cur.Right
		}
	}

	return true
}

func isSymmetric(root *TreeNode) bool {
	return check(root.Left, root.Right)
}

func check(left, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	}

	if left == nil || right == nil {
		return false
	}

	return left.Val == right.Val && check(left.Left, right.Right) && check(left.Right, right.Left)
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}

	stack := make([]*TreeNode, 0)
	stack = append(stack, root)
	count := 1

	for len(stack) > 0 {
		size := len(stack)
		for i := 0; i < size; i++ {
			cur := stack[i]
			if cur.Left != nil {
				stack = append(stack, cur.Left)
			}
			if cur.Right != nil {
				stack = append(stack, cur.Right)
			}
		}
		if size != len(stack) {
			count++
		}
		stack = stack[size:]
	}

	return count
}

func flatten(root *TreeNode) {
	var head, tail *TreeNode
	stack := make([]*TreeNode, 0)
	cur := root

	for cur != nil || len(stack) > 0 {
		if cur != nil {
			if head == nil {
				head = &TreeNode{Val: cur.Val}
				tail = head
			} else {
				tmp := &TreeNode{Val: cur.Val}
				tail.Right = tmp
				tail = tmp
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

//func maxProfit(prices []int) int {
//	min := math.MaxInt64
//	max := 0
//	for i := 0; i < len(prices); i++ {
//		if prices[i] < min {
//			min = prices[i]
//		} else if prices[i]-min > max {
//			max = prices[i] - min
//		}
//	}
//
//	return max
//}
//
//func maxPathSum(root *TreeNode) int {
//	maxSum := math.MinInt32
//	var maxGen func(node *TreeNode) int
//	maxGen = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//
//		leftMax := max(maxGen(node.Left), 0)
//		rightMax := max(maxGen(node.Right), 0)
//
//		sum := node.Val + leftMax + rightMax
//		maxSum = max(sum, maxSum)
//
//		return node.Val + max(leftMax, rightMax)
//	}
//	maxGen(root)
//	return maxSum
//}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func singleNumber(nums []int) int {
	var single int
	for i := 0; i < len(nums); i++ {
		single ^= nums[i]
	}
	return single
}

func hasCycle(head *ListNode) bool {
	hm := make(map[*ListNode]struct{})
	for head != nil {
		if _, ok := hm[head]; ok {
			return true
		}
		hm[head] = struct{}{}
		head = head.Next
	}

	return false
}

//func main() {
//	root := &TreeNode{
//		Val: 1,
//		Left: &TreeNode{
//			Val: 2,
//			Left: &TreeNode{
//				Val: 4,
//			},
//			Right: &TreeNode{
//				Val: 5,
//			},
//		},
//		Right: &TreeNode{
//			Val: 3,
//			Left: &TreeNode{
//				Val: 6,
//			},
//			Right: &TreeNode{
//				Val: 7,
//			},
//		},
//	}
//	flatten(root)
//}
