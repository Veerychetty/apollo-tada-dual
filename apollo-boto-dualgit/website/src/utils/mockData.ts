import { TableData } from '../types';

export const generateMockOCRData = (): TableData => {
  return {
    headers: [
      'Parameter',
      'Top View',
      'Bottom View',
      'Specification',
      'Status',
      'Notes'
    ],
    rows: [
      ['Tyre Width', '205 mm', '205 mm', '205 ±3 mm', 'Pass', 'Within tolerance'],
      ['Aspect Ratio', '55', '55', '55', 'Pass', 'Correct'],
      ['Rim Diameter', '16 inches', '16 inches', '16 inches', 'Pass', 'Standard size'],
      ['Load Index', '91', '91', '91', 'Pass', 'Verified'],
      ['Speed Rating', 'V', 'V', 'V (240 km/h)', 'Pass', 'High performance'],
      ['DOT Code', 'DOT XYZ1 2023', 'DOT XYZ1 2023', 'Valid format', 'Pass', 'Manufacturing week 20, 2023'],
      ['Tread Depth', '8.2 mm', '8.1 mm', '≥7.5 mm', 'Pass', 'Good condition'],
      ['Brand', 'PREMIUM TYRES', 'PREMIUM TYRES', 'Approved', 'Pass', 'Registered brand'],
      ['Model', 'UltraGrip Pro', 'UltraGrip Pro', 'UG-Pro Series', 'Pass', 'Current model'],
      ['Sidewall Condition', 'Good', 'Good', 'No cracks/damage', 'Pass', 'Visual inspection OK'],
      ['Tread Pattern', 'Asymmetric', 'Asymmetric', 'AG-123 Pattern', 'Pass', 'Design verified'],
      ['Max Pressure', '51 PSI', '51 PSI', '51 PSI', 'Pass', '350 kPa'],
    ]
  };
};
