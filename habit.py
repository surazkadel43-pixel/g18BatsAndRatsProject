import pandas as pd

# Example: load your dataset (replace 'dataset1.csv' with actual file path)
df = pd.read_csv('dataset1.csv')

# Mapping habits to categories and descriptions
habit_mapping = {
    # Bat-only
    "fast": ("Bat-only", "Bat moved quickly"),
    "bat": ("Bat-only", "Single bat present"),
    "bats": ("Bat-only", "Multiple bats present"),
    "other_bats": ("Bat-only", "Other bats in the scene"),
    "other_bat": ("Bat-only", "Another bat in the scene"),
    "bat_fight": ("Bat-only", "Bat fighting another bat"),
    "bat_figiht": ("Bat-only", "Bat fighting another bat (typo)"),
    "pick": ("Bat-only", "Bat picking food"),
    "pick_and_others": ("Bat-only", "Bat picking food with others present"),
    "pick_and_all": ("Bat-only", "Bat picking food with all nearby"),
    "all_pick": ("Bat-only", "All bats picking food"),
    "pick_and_bat": ("Bat-only", "Bat picking food while another bat nearby"),
    "pick_bat": ("Bat-only", "Bat picking food"),
    "bat_and_pick": ("Bat-only", "Bat and food picking happening"),
    "bat_and_pick_far": ("Bat-only", "Bat picking food far from something"),
    "bat_pick": ("Bat-only", "Bat picking food"),
    "fast_and_pick": ("Bat-only", "Bat moved fast and picked food"),
    "eating": ("Bat-only", "Bat eating food"),
    "gaze": ("Bat-only", "Bat watching/vigilant"),
    "eating_and_bat_and_pick": ("Bat-only", "Bat eating and picking food"),

    # Rat-only
    "rat": ("Rat-only", "Rat present"),
    "rat_pick": ("Rat-only", "Rat picking food"),
    "pick_and_rat": ("Rat-only", "Food picked while rat present"),
    "pick_rat": ("Rat-only", "Food picked by rat"),
    "rat_and_pick": ("Rat-only", "Rat and picking food at same time"),
    "rat_and_no_food": ("Rat-only", "Rat present but no food"),
    "rat_and_others": ("Rat-only", "Rat and other animals present"),
    "rat_attack": ("Rat-only", "Rat attacking"),
    "attack_rat": ("Rat-only", "Attack on rat"),
    "rat_disappear": ("Rat-only", "Rat disappeared"),
    "rat_and_rat": ("Rat-only", "Multiple rats present"),
    "not_sure_rat": ("Rat-only", "Unclear if rat present"),
    "rat_to_bat": ("Rat-only", "Rat moving toward bat"),

    # Bat–Rat interactions
    "bat_and_rat": ("Bat–Rat interaction", "Bat and rat present together"),
    "rat_and_bat": ("Bat–Rat interaction", "Rat and bat present together"),
    "bat_pick_rat": ("Bat–Rat interaction", "Bat picking food while rat present"),
    "pick_rat_bat": ("Bat–Rat interaction", "Food picking involving rat and bat"),
    "pick_rat_and_bat": ("Bat–Rat interaction", "Food picking with rat and bat present"),
    "rat_pick_and_bat": ("Bat–Rat interaction", "Rat picking food while bat present"),
    "rat_bat": ("Bat–Rat interaction", "Rat and bat together"),
    "rat_bat_fight": ("Bat–Rat interaction", "Fight between bat and rat"),
    "fight_rat": ("Bat–Rat interaction", "Bat fighting rat"),
    "fight_bat": ("Bat–Rat interaction", "Fight involving bat"),
    "bat_fight_and_rat": ("Bat–Rat interaction", "Bat fight while rat present"),
    "bat_fight_rat": ("Bat–Rat interaction", "Bat fighting rat"),
    "bat_rat": ("Bat–Rat interaction", "Bat and rat present"),
    "bat_rat_pick": ("Bat–Rat interaction", "Bat and rat picking food"),
    "pick_bat_rat": ("Bat–Rat interaction", "Food picking involving bat and rat"),
    "eating_bat_rat_pick": ("Bat–Rat interaction", "Eating and picking involving bat and rat"),
    "rat_and_bat_and_pick": ("Bat–Rat interaction", "Rat, bat, and food picking together"),
    "gaze": ("Bat-only", "Bat watching/vigilant"),

    # Other / Special
    "no_food": ("Other", "No food on platform"),
    "bowl_out": ("Other", "Food bowl removed or empty"),
    "other": ("Other", "Other/uncategorised behaviour"),
    "other_directions": ("Other", "Bat moving in other directions"),
    "pup_and_mon": ("Other", "Pup and mother bat"),
    "other_bats/rat": ("Other", "Other bats and rat present"),
}

# Apply mapping
df["habit_category"] = df["habit"].map(lambda x: habit_mapping.get(x, ("Other", "Unmapped/coordinate data"))[0])
df["habit_description"] = df["habit"].map(lambda x: habit_mapping.get(x, ("Other", "Unmapped/coordinate data"))[1])

# Count per category
category_counts = df["habit_category"].value_counts()

print(category_counts)
print(df[["habit", "habit_category", "habit_description"]].head())
