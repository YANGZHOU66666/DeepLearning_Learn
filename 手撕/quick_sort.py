
nums = [0,2,4,2,4,6,3,8]

# l = 0
# target = 3

def quick_sort(l,r):
    if l>=r:
        return
    target = nums[l]
    left = l
    target_l = l
    for i in range(l,r+1):
        if nums[i]<target:
            temp = nums[i]
            nums[i] = nums[left]
            nums[left] = temp
            left += 1
            if nums[i] == target:
                temp = nums[i]
                nums[i] = nums[target_l]
                nums[target_l] = temp
            target_l+=1
        elif nums[i]==target:
            temp = nums[i]
            nums[i] = nums[target_l]
            nums[target_l] = temp
            target_l+=1

    quick_sort(l,left-1)
    quick_sort(target_l,r)

quick_sort(0,len(nums)-1)

'''
for i in range(0,len(nums)):
    if nums[i]<=target:
        temp = nums[i]
        nums[i] = nums[l]
        nums[l] = temp
        l+=1


'''

for i in range(0,len(nums)):
    print(nums[i])