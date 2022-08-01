package main

import "fmt"

type LRULink struct {
	Pre   *LRULink
	Next  *LRULink
	Name  int
	Value int
}

type LRU struct {
	cache map[int]*LRULink
	head  *LRULink
	tail  *LRULink
	cap   int
	used  int
}

func NewLRU(cap int) *LRU {
	return &LRU{
		cache: make(map[int]*LRULink, 0),
		cap:   cap,
	}
}

func (l *LRU) Put(name, value int) {
	link, ok := l.cache[name]
	// 存在
	if ok {
		link.Pre.Next = link.Next
		link.Next.Pre = link.Pre

		head := l.head
		head.Pre = link

		link.Pre = nil
		link.Next = head
		l.head = link

		link.Value = value
		return
	}

	// 不存在
	link = &LRULink{
		Name:  name,
		Value: value,
		Next:  l.head,
	}
	l.cache[name] = link

	// 头部不存在
	if l.head == nil {
		l.head = link
		l.tail = link
		l.used++
		return
	}

	// 头部存在
	head := l.head
	head.Pre = link
	l.head = link

	// 满了
	if l.used == l.cap {
		tail := l.tail
		tail.Pre.Next = nil
		l.tail = tail.Pre

		delete(l.cache, tail.Name)

		return
	}

	// 没满
	l.used++
	return
}

func (l *LRU) Get(name int) (int, bool) {
	link, ok := l.cache[name]
	if !ok {
		return 0, false
	}

	if link.Pre != nil {
		link.Pre.Next = link.Next
	}
	if link.Next != nil {
		link.Next.Pre = link.Pre
	}

	head := l.head
	head.Pre = link
	link.Next = head
	link.Pre = nil
	l.head = link

	return link.Value, true

}

func main() {
	lru := NewLRU(3)

	lru.Put(1, 1)
	lru.Put(2, 1)
	lru.Put(3, 1)

	lru.Put(4, 1)

	lru.Get(2)

	fmt.Println(lru.head)

}
