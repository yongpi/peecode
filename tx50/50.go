package main

import (
	"fmt"
	"math"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var header, tail *ListNode
	carry := 0

	for l1 != nil || l2 != nil {
		sum := carry
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}

		sum, carry = sum%10, sum/10
		if header == nil {
			header = &ListNode{Val: sum}
			tail = header
			continue
		}
		if tail != nil {
			tmp := &ListNode{Val: sum}
			tail.Next = tmp
			tail = tmp
		}
	}

	if carry > 0 && tail != nil {
		tail.Next = &ListNode{Val: carry}
	}

	return header
}

func longestPalindrome(s string) string {
	if s == "" {
		return ""
	}
	var start, end int
	for i := 0; i < len(s); i++ {
		l1, r1 := longest(s, i, i)
		l2, r2 := longest(s, i, i+1)
		if r1-l1 > end-start {
			end, start = r1, l1
		}
		if r2-l2 > end-start {
			end, start = r1, l1
		}
	}

	return s[start : end+1]
}

func longest(s string, left, right int) (int, int) {
	for ; left >= 0 && right < len(s) && s[left] == s[right]; left, right = left-1, right+1 {
	}
	return left + 1, right - 1
}

func reverse(x int) (rev int) {
	for x != 0 {
		if rev < math.MinInt32/10 || rev > math.MaxInt32/10 {
			return 0
		}

		digit := x % 10
		x /= 10
		rev = rev*10 + digit
	}

	return rev
}

func isPalindrome(x int) bool {
	if x < 0 || (x%10 == 0 && x != 0) {
		return false
	}

	var rev int
	for x > rev {
		d := x % 10
		x /= 10
		rev = rev*10 + d
	}

	return x == rev || x == rev/10
}

func lengthOfLongestSubstring(s string) int {
	max, r, n := 0, 0, len(s)
	m := make(map[byte]bool, n)

	for i := 0; i < n; i++ {
		if i > 0 {
			delete(m, s[i-1])
		}

		for ; r < n && !m[s[r]]; r++ {
			m[s[r]] = true
		}

		if max < r-i {
			max = r - i
		}
		if max == n {
			return max
		}
	}

	return max

}

func getK(x, y []int, k int) int {
	var xi, yi int
	for {
		if xi == len(x) {
			return y[yi+k-1]
		}
		if yi == len(y) {
			return x[xi+k-1]
		}
		if k == 1 {
			return min(x[xi], y[yi])
		}

		half := k/2 - 1
		nxi := min(xi+half, len(x)-1)
		nyi := min(yi+half, len(y)-1)
		if x[nxi] <= y[nyi] {
			k -= nxi - xi + 1
			xi = nxi + 1
		} else {
			k -= nyi - yi + 1
			yi = nyi + 1
		}
	}
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	count := len(nums1) + len(nums2)
	k := count / 2
	if count%2 == 0 {
		return float64(getK(nums1, nums2, k)+getK(nums1, nums2, k+1)) / 2.0
	}
	return float64(getK(nums1, nums2, k+1))
}

func maxArea(height []int) int {
	l, r := 0, len(height)-1
	max := 0

	for r > l {
		lv := height[l]
		rv := height[r]
		size := r - l
		var area int
		if lv >= rv {
			area = rv * size
			r--
		} else {
			area = lv * size
			l++
		}
		if area > max {
			max = area
		}
	}

	return max
}

//func threeSum(nums []int) [][]int {
//	count := len(nums)
//	sort.Ints(nums)
//	var res [][]int
//
//	for first := 0; first < count; first++ {
//		if first > 0 && nums[first] == nums[first-1] {
//			continue
//		}
//
//		third := count - 1
//		value := -1 * nums[first]
//
//		for second := first + 1; second < count; second++ {
//			if second > first+1 && nums[second] == nums[second-1] {
//				continue
//			}
//
//			for ; second < third && nums[second]+nums[third] > value; third-- {
//			}
//
//			if second == third {
//				break
//			}
//
//			if nums[second]+nums[third] == value {
//				res = append(res, []int{nums[first], nums[second], nums[third]})
//			}
//		}
//	}
//
//	return res
//}

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

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	var combinations []string
	var trackBack func(string, int, string)
	trackBack = func(digits string, i int, comb string) {
		if len(comb) == len(digits) {
			combinations = append(combinations, comb)
			return
		}

		nums := phoneMap[string(digits[i])]
		for _, value := range nums {
			trackBack(digits, i+1, comb+string(value))
		}
	}

	trackBack(digits, 0, "")
	return combinations
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	first, next := head, dummy
	for i := 0; i < n; i++ {
		first = first.Next
	}
	for ; first != nil; first = first.Next {
		next = next.Next
	}

	next.Next = next.Next.Next
	return dummy.Next
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}

	if list2 == nil {
		return list1
	}

	if list1.Val > list2.Val {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	} else {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	}
}

func generateParenthesis(n int) []string {
	var res []string
	var trackBack func(value string, open, close, max int)
	trackBack = func(value string, open, close, max int) {
		if len(value) == max*2 {
			res = append(res, value)
			return
		}

		if open < max {
			value += "("
			trackBack(value, open+1, close, max)
			value = value[:len(value)-1]
		}
		if open > close {
			value += ")"
			trackBack(value, open, close+1, max)
			value = value[:len(value)-1]
		}
	}

	trackBack("", 0, 0, n)
	return res
}

func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}

	return mergeNode(lists, 0, len(lists)-1)
}

func mergeNode(lists []*ListNode, left, right int) *ListNode {
	if left >= right {
		return lists[left]
	}

	mid := left + (right-left)/2
	l1 := mergeNode(lists, left, mid)
	l2 := mergeNode(lists, mid+1, right)
	return mergeTwoLists(l1, l2)
}

func nextPermutation(nums []int) {
	i, j, k := len(nums)-2, len(nums)-1, len(nums)-1

	for i > 0 && nums[i] >= nums[j] {
		i--
		j--
	}

	if i > 0 {
		for k > i && nums[k] <= nums[i] {
			k--
		}
		nums[i], nums[k] = nums[k], nums[i]
	}

	for m, n := j, len(nums)-1; m < n; m, n = m+1, n-1 {
		nums[m], nums[n] = nums[n], nums[m]
	}
}

func longestValidParentheses(s string) int {
	max := 0
	stack := make([]int, 0)
	stack = append(stack, -1)
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			stack = append(stack, i)
			continue
		}
		stack = stack[:len(stack)-1]
		if len(stack) == 0 {
			stack = append(stack, i)
			continue
		}
		if max < i-stack[len(stack)-1] {
			max = i - stack[len(stack)-1]
		}
	}

	return max
}

func search(nums []int, target int) int {
	start, end := 0, len(nums)-1
	for start < end {
		if nums[start] == target {
			return start
		}
		if nums[end] == target {
			return end
		}

		mid := start + (end-start)/2
		if nums[mid] == target {
			return mid
		}
		if nums[start] < nums[mid] {
			if nums[start] < target && nums[mid] > target {
				end = mid - 1
			} else {
				start = mid + 1
			}
		} else {
			if nums[mid] < target && nums[end] > target {
				start = mid + 1
			} else {
				end = mid - 1
			}
		}
	}
	if nums[start] == target {
		return start
	}

	return -1
}

func searchRange(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}

	index := binaryIndex(nums, target)
	if nums[index] != target {
		return []int{-1, -1}
	}
	j := index
	for i := index + 1; i < len(nums); i++ {
		if nums[i] > target {
			j = i - 1
			break
		}
		j++
	}

	return []int{index, j}
}

func binaryIndex(nums []int, target int) int {
	start, end := 0, len(nums)-1
	for start < end {
		mid := (end + start) / 2
		if nums[mid] < target {
			start = mid + 1
		}
		if nums[mid] >= target {
			end = mid
		}
	}

	return start
}

func permute(nums []int) [][]int {
	var res [][]int
	used := make([]bool, len(nums))
	var values []int
	var trackBack func(nums, values []int, used []bool)
	trackBack = func(nums, values []int, used []bool) {
		if len(values) == len(nums) {
			var tmp []int
			tmp = append(tmp, values...)
			res = append(res, tmp)
			return
		}

		for i := 0; i < len(nums); i++ {
			if !used[i] {
				used[i] = true
				values = append(values, nums[i])
				trackBack(nums, values, used)
				used[i] = false
				values = values[:len(values)-1]
			}
		}
	}
	trackBack(nums, values, used)
	return res
}

func rotate(matrix [][]int) {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-i-1] = matrix[n-i-1], matrix[i]
	}

	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
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
			if move > n-1 {
				return true
			}
		} else {
			return false
		}
	}

	return false
}

func sortForSlice(data [][]int, lo, hi int) {
	if lo >= hi {
		return
	}
	i := position(data, lo, hi)
	sortForSlice(data, lo, i-1)
	sortForSlice(data, i+1, hi)
}

func position(data [][]int, lo, hi int) int {
	cv := data[lo][0]
	i := lo + 1
	j := hi
	for {
		for ; data[i][0] <= cv; i++ {
			if i == hi {
				break
			}
		}
		for ; data[j][0] > cv; j-- {
			if j == lo {
				break
			}
		}
		if i >= j {
			break
		}
		data[i], data[j] = data[j], data[i]
	}

	data[lo], data[j] = data[j], data[lo]
	return j
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func merge(intervals [][]int) [][]int {
	sortForSlice(intervals, 0, len(intervals)-1)

	var result [][]int
	merge := intervals[0]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] >= merge[0] && intervals[i][0] <= merge[1] {
			merge[0] = min(intervals[i][0], merge[0])
			merge[1] = max(merge[1], intervals[i][1])
		} else {
			result = append(result, append([]int(nil), merge...))
			merge = intervals[i]
		}
	}
	result = append(result, merge)

	return result
}

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}

	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}

	return dp[m-1][n-1]
}

func minPathSum(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]

	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}

	for i := 1; i < n; i++ {
		dp[0][i] = dp[0][i-1] + grid[0][i]
	}

	for i := 1; i < m; i++ {
		for j := 0; j < n; j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}

	return dp[m-1][n-1]
}

func climbStairs(n int) int {
	var first, second int

	// 表示 dp[0]
	sum := 1
	for i := 1; i <= n; i++ {
		first = second
		second = sum
		sum = first + second
	}

	return sum
}

func sortColors(nums []int) {
	p0, p1 := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			nums[i], nums[p0] = nums[p0], nums[i]
			if p0 < p1 {
				nums[i], nums[p1] = nums[p1], nums[i]
			}
			p0++
			p1++
		} else if nums[i] == 1 {
			nums[i], nums[p1] = nums[p1], nums[i]
			p1++
		}

	}
}

func contain(a, b map[byte]int) bool {
	for key, value := range b {
		if va, ok := a[key]; !ok || va < value {
			return false
		}
	}

	return true
}
func minWindow(s string, t string) string {
	sm, tm := make(map[byte]int), make(map[byte]int)
	for i := 0; i < len(t); i++ {
		tm[t[i]]++
	}

	sl, sr := -1, -1
	for l, r := 0, 0; r < len(s); r++ {
		sm[s[r]]++
		for ; contain(sm, tm) && l <= r; l++ {
			if sr < 0 || sr-sl > r-l {
				sr, sl = r, l
			}
			sm[s[l]]--
		}
	}
	if sr < 0 {
		return ""
	}

	return s[sl : sr+1]
}

func subsets(nums []int) (ans [][]int) {
	stack := make([]int, 0)
	var trackBack func(nums []int, index int)
	trackBack = func(nums []int, index int) {
		tmp := make([]int, 0)
		tmp = append(tmp, stack...)
		ans = append(ans, tmp)
		for i := index; i < len(nums); i++ {
			stack = append(stack, nums[i])
			trackBack(nums, i+1)
			stack = stack[:len(stack)-1]
		}
	}
	trackBack(nums, 0)
	return ans
}

func subsets2(nums []int) (ans [][]int) {
	mask := 1 << len(nums)
	for i := 0; i < mask; i++ {
		set := make([]int, 0)
		for key, value := range nums {
			if i>>key&1 > 0 {
				set = append(set, value)
			}
		}
		ans = append(ans, set)
	}

	return ans
}

func exist(board [][]byte, word string) bool {
	m, n := len(board), len(board[0])
	em := make([][]bool, m)
	for i := 0; i < m; i++ {
		em[i] = make([]bool, n)
	}

	dic := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	var trackBack func(board [][]byte, x, y, k int) bool
	trackBack = func(board [][]byte, x, y, k int) bool {
		if board[x][y] != word[k] {
			return false
		}
		if k == len(word)-1 {
			return true
		}

		em[x][y] = true
		for _, item := range dic {
			xi, yi := x+item[0], y+item[1]
			if xi >= 0 && xi < m && yi >= 0 && yi < n && !em[xi][yi] {
				if trackBack(board, xi, yi, k+1) {
					return true
				}
			}
		}
		em[x][y] = false
		return false
	}

	for x, row := range board {
		for y, _ := range row {
			if trackBack(board, x, y, 0) {
				return true
			}
		}
	}

	return false
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	stack := make([]*TreeNode, 0)
	stack = append(stack, root)
	res := make([][]int, 0)
	for len(stack) > 0 {
		size := len(stack)
		item := make([]int, size)
		for i := 0; i < size; i++ {
			node := stack[i]
			item[i] = node.Val
			if node.Left != nil {
				stack = append(stack, node.Left)
			}
			if node.Right != nil {
				stack = append(stack, node.Right)
			}
		}
		res = append(res, item)
		stack = stack[size:]
	}
	return res
}

//func buildTree(preorder []int, inorder []int) *TreeNode {
//	if len(preorder) == 0 {
//		return nil
//	}
//	root := &TreeNode{Val: preorder[0]}
//	var i int
//	for ; i < len(inorder); i++ {
//		if inorder[i] == preorder[0] {
//			break
//		}
//	}
//
//	root.Left = buildTree(preorder[1:i+1], inorder[:i])
//	root.Right = buildTree(preorder[i+1:], inorder[i+1:])
//	return root
//}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}

	stack := make([]*TreeNode, 0)
	root := &TreeNode{Val: preorder[0]}
	stack = append(stack, root)
	var ii int
	for j := 1; j < len(preorder); j++ {
		node := stack[len(stack)-1]
		if node.Val != inorder[ii] {
			node.Left = &TreeNode{Val: preorder[j]}
			stack = append(stack, node.Left)
		} else {
			for ; len(stack) > 0 && stack[len(stack)-1].Val == inorder[ii]; ii++ {
				node = stack[len(stack)-1]
				stack = stack[:len(stack)-1]
			}
			node.Right = &TreeNode{Val: preorder[j]}
			stack = append(stack, node.Right)
		}
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

func maxPathSum(root *TreeNode) int {
	maxGen := math.MinInt64
	var gen func(node *TreeNode) int
	gen = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		left := max(gen(node.Left), 0)
		right := max(gen(node.Right), 0)
		sum := node.Val + left + right
		maxGen = max(maxGen, sum)
		return node.Val + max(left, right)
	}

	gen(root)
	return maxGen
}

func longestConsecutive(nums []int) int {
	hm := make(map[int]bool)
	for _, value := range nums {
		hm[value] = true
	}

	longest := 0
	for num, _ := range hm {
		if !hm[num-1] {
			curl := 1
			for ; hm[num+1]; num++ {
				curl++
			}
			if curl > longest {
				longest = curl
			}
		}
	}

	return longest
}

//func wordBreak(s string, wordDict []string) bool {
//	bk := false
//	var trackBack func(s, value string, wordDict []string)
//	trackBack = func(s, value string, wordDict []string) {
//		if bk {
//			return
//		}
//		if len(value) >= len(s) {
//			if value == s {
//				bk = true
//			}
//			return
//		}
//
//		for _, word := range wordDict {
//			trackBack(s, value+word, wordDict)
//			if bk {
//				return
//			}
//		}
//	}
//	trackBack(s, "", wordDict)
//	return bk
//}

func wordBreak(s string, wordDict []string) bool {
	hm := make(map[string]bool)
	for _, word := range wordDict {
		hm[word] = true
	}

	dp := make([]bool, len(s)+1)
	dp[0] = true

	for i := 1; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			if dp[j] && hm[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}

	return dp[len(s)]
}

func hasCycle(head *ListNode) bool {
	if head == nil {
		return false
	}

	slow, fast := head, head.Next
	for slow != fast {
		if fast == nil || fast.Next == nil {
			return false
		}
		slow = slow.Next
		fast = fast.Next.Next
	}

	return true
}

func main() {
	fmt.Println(wordBreak("catsandog", []string{"cats", "dog", "sand", "and", "cat"}))
}
