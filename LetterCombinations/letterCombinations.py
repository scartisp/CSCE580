class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        digitMap = {
            '2':  ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        ans = []
        for char in digitMap[digits[0]]:
            ans.append(char)
        digits = digits[1:]

        for digit in digits:
            temp = []
            for string in ans:
                for char in digitMap[digit]:
                    temp.append(string+char)
            ans = temp
        return ans
