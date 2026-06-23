import pandas as pd
import os

# Define file paths
base_dir = os.path.expanduser("~/research/ELP_Research/GunshotDetector/data")
input_csv = os.path.join(base_dir, "manifest.csv")
output_csv = os.path.join(base_dir, "model2_threat_manifest.csv")

print(f"Loading master manifest: {input_csv}")
df = pd.read_csv(input_csv)

# Step 1: Isolate true gunshots (Drop ambient/urban sounds where label is not 'gunshot')
df_gunshots = df[df['label'] == 'gunshot'].copy()
print(f"Filtered out ambient noise. Total gunshot samples found: {len(df_gunshots)}")

# Step 2: Define explicit mapping lists matching your datasets exactly
low_threat_models = [
    # Handguns, Pistols, Revolvers, and .22LR Calibers
    '38sws-dot38-caliber', 'beretta-92', 'colt-1911', 'desert-eagle', 
    'glock', 'glock-18c', 'glock-19-9mm-luger-pistol', 'high-standard-22lr', 
    'hk-usp-compact-40-sw-pistol', 'kimber-45acp', 'lorcin-380acp', 'nagant-m1895', 
    'remington-22lr', 'rhino-60ds', 'ruger-22lr', 'ruger-357', 
    'sw-10-8-38spl-revolver', 'sw-34-1-22lr-revolver', 'sig-p225',
    
    # Submachine Guns (SMGs) / Personal Defense Weapons (PDWs)
    'fn-p90', 'hk-ump45', 'kriss-vector', 'mp-40-40-sw-pistol', 
    'mp5-smg', 'pp-19-bizon', 'thompson-m1928', 'uzi-smg',
    
    # Emrah List Handguns & SMGs (including raw matching variants)
    'imi desert eagle (desert eagle)', 'imi desert eagle', 'mp5',
    
    # Fallback categories and generic signatures
    'urban_gunshot', 'unknown_noisy'
]

high_threat_models = [
    # Assault Rifles, Battle Rifles, and Carbines
    'ak-12', 'ak-47', 'daewoo-k2', 'fn-scar', 'hk-g36c', 'm16', 'm4', 
    'mini-14', 'mk14-ebr', 'ots-14-groza', 'qbz-95', 'ruger-ar-556', 
    'sks-rifle', 'slr-rifle', 'steyr-aug', 'vss-vintorez', 'winchester-m14',
    
    # Sniper Rifles / Bolt-Action Rifles
    'arctic-warfare-magnum', 'kar98k', 'm24-sws', 'qbu-88', 'remington-700', 
    'winchester-rifle',
    
    # Shotguns
    'double-barrel-shotgun', 'pump-action-shotgun', 'remington-870', 'saiga-12k',
    
    # Light Machine Guns (LMGs) / Machine Guns
    'dp-27-lmg', 'm249-lmg', 'mg42-lmg', 'vector-lmg',
    
    # Emrah List Rifles and Machine Guns
    'm249', 'mg-42', 'zastava m92'
]

# Mapping function using exact matching against standardized strings
def map_firearm_to_threat(gun_type):
    # Standardize string formatting to ensure reliable matching
    gun_str = str(gun_type).lower().strip()
    
    if gun_str in high_threat_models:
        return 1
    elif gun_str in low_threat_models:
        return 0
    else:
        # Fallback tracking if something slips past the list formatting
        return -1

# Apply the strict mapping configuration
df_gunshots['threat_level'] = df_gunshots['gun_type'].apply(map_firearm_to_threat)

# Step 3: Verify and handle unmapped entries dynamically
unmapped = df_gunshots[df_gunshots['threat_level'] == -1]
if not unmapped.empty:
    print(f"WARNING: Found {len(unmapped)} rows with unmapped firearm strings!")
    print("Unique unmapped strings found:", unmapped['gun_type'].unique())
    print("Defaulting unmapped gunshots to Class 0 (Low Threat) for pipeline continuity...")
    df_gunshots.loc[df_gunshots['threat_level'] == -1, 'threat_level'] = 0

# Step 4: Write out your dedicated training manifest
df_final = df_gunshots[df_gunshots['threat_level'].isin([0, 1])]
df_final.to_csv(output_csv, index=False)

print(f"Relabeling complete! Saved data to: {output_csv}")

# Output target balances for analysis
distribution = df_final['threat_level'].value_counts()
print("\n=== Model 2 Target Distribution ===")
print(f"Class 0 (Low Threat - Handguns/SMGs):   {distribution.get(0, 0)} samples")
print(f"Class 1 (High Threat - Rifles/Shotguns): {distribution.get(1, 0)} samples")
print("====================================")