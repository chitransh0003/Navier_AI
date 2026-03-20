export const CHEMICAL_PROFILES: Record<string, { density: number; waveSpeed: number; bulkModulus: string; viscosityRef: string }> = {
  MS:      { density: 745,  waveSpeed: 1050, bulkModulus: "1.32 GPa", viscosityRef: "0.55 cSt" },
  HSD:     { density: 840,  waveSpeed: 1320, bulkModulus: "1.48 GPa", viscosityRef: "4.2 cSt"  },
  ATF:     { density: 800,  waveSpeed: 1200, bulkModulus: "1.42 GPa", viscosityRef: "1.5 cSt"  },
  BENZENE: { density: 879,  waveSpeed: 1300, bulkModulus: "1.50 GPa", viscosityRef: "0.65 cSt" },
  LPG:     { density: 508,  waveSpeed: 700,  bulkModulus: "0.43 GPa", viscosityRef: "0.11 cSt" },
};

export type FluidType = keyof typeof CHEMICAL_PROFILES;

export const FLUID_TYPES: FluidType[] = ['MS', 'HSD', 'ATF', 'BENZENE', 'LPG'];
