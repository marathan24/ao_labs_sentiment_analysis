# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Thank you for your curiosity!

Arch file for sentiment analysis
"""

import ao_arch as ar

description = "Sentiment Analysis System"

# Define the architecture layers
arch_i = [8] * 128  # 128 inputs, each 8 bits
arch_z = [2]         # 4 outputs: Positive, Negative, Neutral, No Sentiment
arch_c = []          # No additional connectors

connector_function = "full_conn"

# Initialize the architecture
arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)