mydict = {
    "key1": ["value1", "value2", "value3"],
    "key2": ["value4", "value5", "value6"],
    "key3": ["value7", "value8", "value9"]
}

mydict["key4"] = ["value10", "value11", "value12"]  # Adding a new key-value pair

mydict["key4"][1] = "value13"

# access by key name dict["keyname"] but if there is not key name found it is gonna crash
print(mydict["key1"])


# # so better way is to use the .get() method
# print(mydict.get("key4", "default_value")) # this will not crash

# print(mydict.get("key3", "default_value")) # this will return the value of key3

# # to see all the columns in the dict
# print(mydict.keys())

# # and to see all the values in a dict
# print(mydict.values())

# # The items() method will return each item in a dictionary, as tuples in a list.
# print(mydict.items())

# # check if a key exists
# print("key1" in mydict) # True
# print("key4" in mydict) # False

# # check if a value exists
# print("value1" in mydict.values()) # True
# print("value4" in mydict.values()) # False

###################################################################################################

# now how to add new keys and values to the key

mydict["key5"] = ["value14", "value15", "value16"]  # Adding a new key-value pair

#adding multiples keys at once

mydict.update({
    "numbers": [1, 2, 3],
    "letters": ["a", "b", "c"]
})
# adding values to key5
mydict["key5"].append("value17")
mydict["key5"].append("value18")

# another ways
mydict["key5"] += ["value19", "value20"]

#adding one or more value
mydict["key5"] += ["value21", "value22"]


##################################################################################################

# now how to remove the values and key from the dict

#first of all removing the key
del mydict["key5"] # but throws error if not found

# remove the key using pop() and avoid error if not found
mydict.pop("key4", None)

# remove value from the key
mydict["key3"].remove("value7")

#remove all values from a key but keep the key also return error if key not found
mydict["key4"].clear()




#################################################################################

#changing the value of a key but keep the key
mydict["key4"] = ["value13", "value14", "value15"] #change the value of existing keys


#update the value inside the list
mydict["key4"][0] = "value20"

# changing the keys
mydict["key5"] = mydict.pop("key4") 

# using update()
mydict.update({"key5": ["value13", "value14", "value15"]})  # this will update if key exist
# if the keys do not exist it make the new key pair value