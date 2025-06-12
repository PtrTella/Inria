# Definizione dei layer convoluzionali nei primi 3 blocchi della ResNet-18 come da documento

# Ogni entry è (layer, H_out, W_out, K, C_in, C_out, Num_Conv_Layer)
# Per semplificare, useremo solo il conteggio delle convoluzioni principali

layers = [
    # Layer 1 - Conv2d: [32, 64, 112, 112]
    ("Conv1", 112, 112, 7, 3, 64, 1),  # Assume input RGB = 3 channels, kernel size 7
    # Layer 2 - BasicBlock x2: [32, 64, 56, 56]
    ("Block1_Conv1", 56, 56, 3, 64, 64, 1),
    ("Block1_Conv2", 56, 56, 3, 64, 64, 1),
    ("Block2_Conv1", 56, 56, 3, 64, 64, 1),
    ("Block2_Conv2", 56, 56, 3, 64, 64, 1),
    # Layer 3 - BasicBlock x2: [32, 128, 28, 28]
    ("Block3_Conv1", 28, 28, 3, 64, 128, 1),
    ("Block3_Conv2", 28, 28, 3, 128, 128, 1),
    ("Block4_Conv1", 28, 28, 3, 128, 128, 1),
    ("Block4_Conv2", 28, 28, 3, 128, 128, 1),
    # Layer 4 - BasicBlock x2: [32, 256, 14, 14]
    ("Block5_Conv1", 14, 14, 3, 128, 256, 1),
    ("Block5_Conv2", 14, 14, 3, 256, 256, 1),
    ("Block6_Conv1", 14, 14, 3, 256, 256, 1),
    ("Block6_Conv2", 14, 14, 3, 256, 256, 1),
]

# Calcolo FLOPs per ciascuna convoluzione: 2 * H * W * K^2 * C_in * C_out
results = []

for name, H, W, K, C_in, C_out, n_convs in layers:
    flops = 2 * H * W * (K ** 2) * C_in * C_out * n_convs
    results.append((name, flops))

# Print results avoiding dataframe for simplicity
for name, flops in results:
    # Print the name and the calculated GFLOPs
    print(f"{name}: {flops / 1e9:.4f} GFLOPs")
# Output the total GFLOPs
total_flops = sum(flops for _, flops in results)
print(f"Total GFLOPs: {total_flops / 1e9:.4f} GFLOPs")



# Assumiamo che solo il 20% dei canali possa essere memoizzato.
# Quindi, per ogni layer, il risparmio teorico è proporzionale a 20% dei canali output (FLOPs evitate)
# Lo student model (Ds) ha metà dei blocchi, stimiamo il suo costo come 50% del totale

fraction_cached = 0.2
fraction_uncached = 1.0 - fraction_cached
student_model_cost = 0.5 * total_flops  # Ds è metà più leggero secondo il paper

# FLOPs effettive dopo caching:
# Si computano solo l'80% delle operazioni dei layer convoluzionali, più il costo del modello student
flops_after_caching = fraction_uncached * total_flops + student_model_cost

# Risparmio netto
flops_saved = total_flops - flops_after_caching
print("\n --- Caching Analysis in optimistic scenario ---")
print(f"Total FLOPs before caching: {float(total_flops) / 1e9:.4f} GFLOPs")
print(f"FLOPs of cached layers: {float(fraction_uncached * total_flops) / 1e9:.4f} GFLOPs ({-fraction_cached * 100:.2f}% of total FLOPs)")
print(f"Total FLOPs after caching: {float(flops_after_caching) / 1e9:.4f} GFLOPs, which is {float(flops_after_caching) / float(total_flops) * 100:.2f}% of the total FLOPs")
print(f"--- Considering the sum of the caching optimization {float(fraction_uncached * total_flops) / 1e9:.4f} and the student model cost {float(student_model_cost) / 1e9:.4f}")
print(f"FLOPs saved: {float(flops_saved) / 1e9:.4f} GFLOPs, which is {float(flops_saved) / float(total_flops) * 100:.2f}% of the total FLOPs")


print("\n --- Caching Analysis in realistic scenario ---")
# In uno scenario pessimista, si assume che il modello principale non possa essere ottimizato per canale e che quindi 
# A meno che non si usi il 100% dei canali, il risparmio è nullo.

flops_after_caching_pessimistic = total_flops + student_model_cost
flops_saved_pessimistic = total_flops - flops_after_caching_pessimistic
print(f"Total FLOPs before caching: {float(total_flops) / 1e9:.4f} GFLOPs")
print(f"FLOPs of cached layers: {total_flops / 1e9:.4f} GFLOPs (no caching)")
print(f"Total FLOPs after caching: {flops_after_caching_pessimistic / 1e9:.4f} GFLOPs, which is {flops_after_caching_pessimistic / total_flops * 100:.2f}% of the total FLOPs")
print(f"--- Considering the sum of the caching optimization {total_flops / 1e9:.4f} and the student model cost {student_model_cost / 1e9:.4f}")
print(f"FLOPs saved: {flops_saved_pessimistic / 1e9:.4f} GFLOPs, which is {flops_saved_pessimistic / total_flops * 100:.2f}% of the total FLOPs")
# In uno scenario pessimista, il risparmio è nullo poiché non si può ottimizzare per canale
# e il modello student non ha un costo inferiore.
# Inoltre, si assume che il modello student non possa essere ottimizzato per canale
# e quindi non si ottiene alcun risparmio.


# Caching in uno scenario ottimistico dove posso fare caching dell'intero layer (100% dei canali)
fraction_cached_optimistic = 1.0
fraction_uncached_optimistic = 1.0 - fraction_cached_optimistic
flops_after_caching_optimistic = fraction_uncached_optimistic * total_flops + student_model_cost

flops_saved_optimistic = total_flops - flops_after_caching_optimistic
print("\n --- Caching Analysis in optimistic scenario with full layer caching ---")
print(f"Total FLOPs before caching: {float(total_flops) / 1e9:.4f} GFLOPs")
print(f"FLOPs of cached layers: {float(fraction_uncached_optimistic * total_flops) / 1e9:.4f} GFLOPs ({-fraction_cached_optimistic * 100:.2f}% of total FLOPs)")
print(f"Total FLOPs after caching: {float(flops_after_caching_optimistic) / 1e9:.4f} GFLOPs, which is {float(flops_after_caching_optimistic) / float(total_flops) * 100:.2f}% of the total FLOPs")
print(f"--- Considering the sum of the caching optimization {float(fraction_uncached_optimistic * total_flops) / 1e9:.4f} and the student model cost {float(student_model_cost) / 1e9:.4f}")
print(f"FLOPs saved: {float(flops_saved_optimistic) / 1e9:.4f} GFLOPs, which is {float(flops_saved_optimistic) / float(total_flops) * 100:.2f}% of the total FLOPs")



