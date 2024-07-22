gpt3_xl = {'l': 2048, 'e': 2048, 'h': 16, 'depth': 24}
gpt3 = {'l': 2048, 'e': 12288, 'h': 96, 'depth': 96}
gpt3_1T = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 128}
gpt3_3T = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 128*3}
vit_era5 = {'l': 64800*4, 'e': 4096, 'h': 64, 'depth': 32}

models = {'gpt3_xl': gpt3_xl,
          'gpt3': gpt3,
          'gpt3_1T': gpt3_1T,
          'gpt3_3T': gpt3_3T,
          'vit_era5': vit_era5}
