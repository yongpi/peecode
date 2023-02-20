package main

import "fmt"

type MinStack struct {
	data []int
	min  []int
}

func NewMinStack() *MinStack {
	return &MinStack{
		data: make([]int, 0),
		min:  make([]int, 0),
	}
}

func (s *MinStack) Push(value int) {
	s.data = append(s.data, value)
	if len(s.min) == 0 {
		s.min = append(s.min, value)
		return
	}

	if value < s.min[len(s.min)-1] {
		s.min = append(s.min, value)
		return
	}

	s.min = append(s.min, s.min[len(s.min)-1])
}

func (s *MinStack) Pop() int {
	value := s.data[len(s.data)-1]
	s.data = s.data[:len(s.data)-1]
	s.min = s.min[:len(s.min)-1]

	return value
}

func (s *MinStack) GetMin() int {
	return s.min[len(s.min)-1]
}

func canPop(in, out []int) bool {
	stack := make([]int, 0)
	oi := 0

	for _, value := range in {
		stack = append(stack, value)
		if value != out[oi] {
			continue
		}

		for len(stack) > 0 && oi < len(out) {
			item := stack[len(stack)-1]
			if item != out[oi] {
				break
			}
			stack = stack[:len(stack)-1]
			oi++
		}
	}

	return oi == len(out)
}

func rain2(rain []int) int {
	var stack []int
	var ans int
	for i := 0; i < len(rain); i++ {
		for len(stack) > 0 && rain[stack[len(stack)-1]] < rain[i] {
			ci := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				break
			}

			ri := stack[len(stack)-1]
			mh := min(rain[ri], rain[i])
			width := i - ri - 1

			ans += (mh - rain[ci]) * width
		}

		stack = append(stack, i)
	}

	return ans
}

func main() {
	//stack := NewMinStack()
	//stack.Push(10)
	//stack.Push(1)
	//stack.Push(5)
	//stack.Push(20)
	//
	//fmt.Println(stack.GetMin())
	//
	//stack.Pop()
	//stack.Pop()
	//stack.Pop()
	//
	//fmt.Println(stack.GetMin())

	fmt.Println(canPop([]int{1, 2, 3, 4, 5}, []int{4, 3, 5, 1, 2}))
}
