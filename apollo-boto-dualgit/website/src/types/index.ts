export interface UploadedImage {
  file: File;
  preview: string;
  type: 'top' | 'bottom';
}

export interface ExtractedCell {
  value: string;
  rowIndex: number;
  colIndex: number;
  isEditing?: boolean;
}

export interface TableData {
  headers: string[];
  rows: string[][];
}

export interface Recipe {
  id: string;
  name: string;
  timestamp: number;
  data: TableData;
  images?: {
    top: string;
    bottom: string;
  };
}

export type AppStep = 'upload' | 'display' | 'extract';
