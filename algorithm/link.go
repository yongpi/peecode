package main

type Link struct {
	Value  int
	Next   *Link
	Pre    *Link
	Random *Link
}

func sortLink(root *Link) *Link {
	return upDownSortLink(root, nil)
}

func upDownSortLink(head, tail *Link) *Link {
	if head == nil {
		return head
	}
	if head.Next == tail {
		head.Next = nil
		return head
	}

	slow := head
	fast := head
	for fast != tail {
		slow = slow.Next
		fast = fast.Next
		if fast != tail {
			fast = fast.Next
		}
	}

	mid := slow
	return mergeLink(upDownSortLink(head, mid), upDownSortLink(mid, tail))
}

func downUpSortLink(head *Link) *Link {
	if head == nil || head.Next == nil {
		return head
	}

	size := 1
	for node := head; node.Next != nil; size++ {
		node = node.Next
	}

	dummy := &Link{Next: head}
	for subLen := 1; subLen < size; subLen = subLen * 2 {
		pre, cur := dummy, dummy.Next
		for cur != nil {
			head1 := cur
			for i := 1; i < subLen && cur.Next != nil; i++ {
				cur = cur.Next
			}

			head2 := cur.Next
			cur.Next = nil
			cur = head2

			for i := 1; i < subLen && cur != nil && cur.Next != nil; i++ {
				cur = cur.Next
			}

			if cur != nil {
				next := cur.Next
				cur.Next = nil
				cur = next
			}

			pre.Next = mergeLink(head1, head2)
			for pre.Next != nil {
				pre = pre.Next
			}
		}
	}

	return dummy.Next
}

func mergeLink(left, right *Link) *Link {
	dummy := &Link{}
	tmp := dummy
	for left != nil && right != nil {
		if left.Value > right.Value {
			tmp.Next = right
			right = right.Next
		} else {
			tmp.Next = left
			left = left.Next
		}

		tmp = tmp.Next
	}

	if left == nil {
		tmp.Next = right
	}
	if right == nil {
		tmp.Next = left
	}

	return dummy.Next
}

func meetLink(left, right *Link) *Link {
	if left == nil || right == nil {
		return nil
	}

	tmp1 := left
	tmp2 := right
	for tmp1 != tmp2 {
		if tmp1 != nil {
			tmp1 = tmp1.Next
		} else {
			tmp1 = right
		}

		if tmp2 != nil {
			tmp2 = tmp2.Next
		} else {
			tmp2 = left
		}
	}

	return tmp1
}

func revertLink(root *Link) *Link {
	var pre *Link
	cur := root

	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}

	return pre
}

func hwLink(head *Link) bool {
	if head == nil || head.Next == nil {
		return false
	}

	slow := head
	fast := head
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	mid := slow
	rl := revertLink(mid.Next)
	ch := head
	for rl != nil {
		if rl.Value != ch.Value {
			return false
		}
		rl = rl.Next
		ch = ch.Next
	}

	return true
}

func copyComplexLink(root *Link) *Link {
	hm := make(map[*Link]*Link)

	var copyLink func(node *Link) *Link
	copyLink = func(node *Link) *Link {
		if node == nil {
			return nil
		}

		if nl, ok := hm[node]; ok {
			return nl
		}

		nn := &Link{Value: node.Value}
		hm[node] = nn

		nn.Next = copyLink(node.Next)
		nn.Random = copyLink(node.Random)

		return nn
	}

	return copyLink(root)
}

func reorderList(head *Link) {
	// 找中间点
	slow, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	sn := slow.Next
	slow.Next = nil

	// 反转
	cur := sn
	var pre *Link
	for cur != nil {
		cn := cur.Next
		cur.Next = pre
		pre = cur
		cur = cn
	}

	n := head
	for n != nil && pre != nil {
		ln, rn := n.Next, pre.Next

		n.Next = pre
		pre.Next = ln

		n = ln
		pre = rn
	}

}

func main() {
	link := &Link{Value: 1, Next: &Link{Value: 2, Next: &Link{Value: 3, Next: &Link{Value: 4}}}}
	//link := &Link{Value: 9, Next: &Link{Value: 3}}
	reorderList(link)

	//fmt.Println(reorderList(link))
}
