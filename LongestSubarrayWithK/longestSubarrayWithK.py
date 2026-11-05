class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        
        left = 0
        myDict = {}
        myDict[nums[left]] = 1
        mostFrequent = 1
        longest = 1
        for i in range(1, len(nums)):
            num = nums[i]
            myDict[num] = myDict.get(num, 0) + 1
            mostFrequent = max(mostFrequent, myDict[num])
            if mostFrequent > k:
                while nums[left] != num:
                    myDict[nums[left]] -= 1
                    left += 1
                myDict[nums[left]] -= 1
                left += 1
                mostFrequent = k

            longest = max(longest, i-left+1)
        
        return longest
