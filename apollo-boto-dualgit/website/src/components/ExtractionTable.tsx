import { useState, useEffect } from 'react';
import { Edit3, Lock, Plus, Trash2, Undo, Redo, X } from 'lucide-react';
import { TableData } from '../types';

interface ExtractionTableProps {
  data: TableData;
  onDataChange: (data: TableData) => void;
}

export default function ExtractionTable({
  data,
  onDataChange,
}: ExtractionTableProps) {
  const [isEditMode, setIsEditMode] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [password, setPassword] = useState('');
  const [selectedCells, setSelectedCells] = useState<number[]>([]);
  const [mergeMode, setMergeMode] = useState(false);
  const [history, setHistory] = useState<any[]>([[]]);
  const [historyIndex, setHistoryIndex] = useState(0);

  // Initialize history when data changes
  useEffect(() => {
    if (data.rows.length > 0 && history.length === 1 && history[0].length === 0) {
      // Store the full table data structure in history
      const tableData = {
        headers: data.headers,
        rows: data.rows.map(row => [...row]) // Deep copy of rows
      };
      setHistory([tableData]);
      setHistoryIndex(0);
    }
  }, [data.rows]);

  const handleAuthSubmit = () => {
    if (password === 'admin123') {
      setIsEditMode(true);
      setShowAuthModal(false);
      setPassword('');
    } else {
      alert('Incorrect password');
    }
  };

  const handleCellClick = (index: number) => {
    if (!isEditMode) return;

    if (mergeMode) {
      if (selectedCells.includes(index)) {
        setSelectedCells(selectedCells.filter(i => i !== index));
      } else {
        setSelectedCells([...selectedCells, index]);
      }
    }
  };

  const handleMerge = () => {
    if (selectedCells.length < 2) {
      alert("Please select at least 2 cells to merge");
      return;
    }

    // Sort selected cells in ascending order to maintain order
    const sortedCells = [...selectedCells].sort((a, b) => a - b);
    
    // Get the text from the DATA column (column 1) of selected rows
    const mergedText = sortedCells
      .map(index => {
        const row = data.rows[index];
        // Get data from column 1 (the actual data column, not serial number)
        return row && row[1] ? row[1].trim() : "";
      })
      .filter(text => text !== "") // Remove empty values
      .join(" "); // Join with single space

    // Create new rows array
    const newRows = [...data.rows];
    
    // Update the FIRST selected row with merged text in DATA column (column 1)
    // Keep the serial number in column 0 unchanged
    if (newRows[sortedCells[0]]) {
      newRows[sortedCells[0]][1] = mergedText;
    }
    
    // Remove other selected rows (from end to start to maintain correct indices)
    for (let i = sortedCells.length - 1; i > 0; i--) {
      newRows.splice(sortedCells[i], 1);
    }

    const newTableData = { ...data, rows: newRows };
    onDataChange(newTableData);
    setSelectedCells([]);
    setMergeMode(false);

    // Save full table data to history
    const tableData = {
      headers: newTableData.headers,
      rows: newTableData.rows.map(row => [...row]) // Deep copy of rows
    };
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(tableData);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      const previousState = history[newIndex];
      
      // If previousState is a table data structure, use it directly
      if (previousState && previousState.headers && previousState.rows) {
        onDataChange(previousState);
      } else {
        // Fallback for old format (should not happen with new implementation)
        const newRows = previousState.map((item: any) => {
          const row = new Array(data.headers.length).fill('');
          row[0] = item.text || '';
          return row;
        });
        
        const newTableData = { ...data, rows: newRows };
        onDataChange(newTableData);
      }
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      const nextState = history[newIndex];
      
      // If nextState is a table data structure, use it directly
      if (nextState && nextState.headers && nextState.rows) {
        onDataChange(nextState);
      } else {
        // Fallback for old format (should not happen with new implementation)
        const newRows = nextState.map((item: any) => {
          const row = new Array(data.headers.length).fill('');
          row[0] = item.text || '';
          return row;
        });
        
        const newTableData = { ...data, rows: newRows };
        onDataChange(newTableData);
      }
    }
  };


  // Add row function
  const handleAddRow = () => {
    const newRow = new Array(data.headers.length).fill('');
    const newRows = [...data.rows, newRow];
    const newData = { ...data, rows: newRows };
    onDataChange(newData);
    
    // Save full table data to history
    const tableData = {
      headers: newData.headers,
      rows: newData.rows.map(row => [...row]) // Deep copy of rows
    };
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(tableData);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  // Remove row function
  const handleRemoveRow = (index: number) => {
    if (data.rows.length > 1) {
      const newRows = data.rows.filter((_, i) => i !== index);
      const newData = { ...data, rows: newRows };
      onDataChange(newData);
      
      // Save full table data to history
      const tableData = {
        headers: newData.headers,
        rows: newData.rows.map(row => [...row]) // Deep copy of rows
      };
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push(tableData);
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
  };

  // Update text function with history tracking
  const handleTextChange = (rowIndex: number, colIndex: number, newText: string) => {
    const newRows = [...data.rows];
    newRows[rowIndex][colIndex] = newText;
    const newData = { ...data, rows: newRows };
    onDataChange(newData);
    
    // Save full table data to history
    const tableData = {
      headers: newData.headers,
      rows: newData.rows.map(row => [...row]) // Deep copy of rows
    };
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(tableData);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };



  const handleReset = () => {
    const resetData = {
      headers: data.headers,
      rows: [new Array(data.headers.length).fill('')]
    };
    onDataChange(resetData);
    setSelectedCells([]);
    setMergeMode(false);
    
    // Reset history with initial empty state
    const initialTableData = {
      headers: resetData.headers,
      rows: resetData.rows.map(row => [...row]) // Deep copy of rows
    };
    setHistory([initialTableData]);
    setHistoryIndex(0);
  };

  const startMergeMode = () => {
    setMergeMode(true);
    setSelectedCells([]);
  };

  const handleSaveChanges = () => {
    setIsEditMode(false);
    setMergeMode(false);
    setSelectedCells([]);
  };

  const isCellSelected = (rowIdx: number) => selectedCells.includes(rowIdx);

  return (
    <div className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Extracted Data
            </h1>
            <p className="text-gray-600">
              Review and edit the extracted tyre mould information
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => setShowAuthModal(true)}
              disabled={isEditMode}
              className={`px-5 py-3 rounded-lg font-medium flex items-center gap-2 transition-all ${
                isEditMode
                  ? 'bg-gray-800 text-white cursor-not-allowed'
                  : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
              }`}
            >
              {isEditMode ? <Edit3 className="w-5 h-5" /> : <Lock className="w-5 h-5" />}
              {isEditMode ? 'Edit Mode Active' : 'Enable Editing'}
            </button>
          </div>
        </div>

        {isEditMode && (
          <div className="mb-6 space-y-4">
            {/* Main Action Buttons */}
            <div className="flex gap-3 flex-wrap">
              <button
                onClick={handleAddRow}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-all shadow-sm"
              >
                <Plus className="w-4 h-4" />
                Add Row
              </button>
              <button
                onClick={handleUndo}
                disabled={historyIndex === 0}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-all shadow-sm disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Undo className="w-4 h-4" />
                Undo
              </button>
              <button
                onClick={handleRedo}
                disabled={historyIndex === history.length - 1}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-all shadow-sm disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Redo className="w-4 h-4" />
                Redo
              </button>
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg font-medium hover:bg-slate-600 transition-all shadow-sm"
              >
                <X className="w-4 h-4" />
                Reset
              </button>
              <button
                onClick={startMergeMode}
                disabled={mergeMode}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-all shadow-sm disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Plus className="w-4 h-4" />
                Merge Cells
              </button>
              <button
                onClick={handleSaveChanges}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-all shadow-sm"
              >
                <Lock className="w-4 h-4" />
                Save Changes
              </button>
            </div>

            {/* Simple Merge Mode - Show confirm button when cells are selected */}
            {mergeMode && selectedCells.length >= 2 && (
              <div className="flex items-center justify-between p-3 bg-gray-100 rounded-lg border border-gray-300">
                <span className="text-sm text-gray-700 font-medium">
                  {selectedCells.length} rows selected
                </span>
                <button
                  onClick={handleMerge}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-all shadow-sm"
                >
                  <Plus className="w-4 h-4" />
                  Confirm Merge ({selectedCells.length})
                </button>
              </div>
            )}
          </div>
        )}

        {/* Table with Scrolling */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          {/* Scrollable Table Container - max height with vertical scroll */}
          <div className="overflow-x-auto overflow-y-auto max-h-[500px]">
            <table className="w-full">
              <thead className="bg-gray-800 sticky top-0 z-10">
                <tr>
                  {isEditMode && (
                    <th className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                      Actions
                    </th>
                  )}
                  {data.headers.map((header, index) => (
                    <th
                      key={index}
                      className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {data.rows.map((row, rowIndex) => (
                  <tr
                    key={rowIndex}
                    className={`hover:bg-gray-50 transition-colors duration-150 ${
                      isEditMode && mergeMode && isCellSelected(rowIndex) ? 'bg-blue-100' : ''
                    } ${rowIndex % 2 === 0 ? 'bg-gray-50' : 'bg-white'}`}
                    onClick={() => handleCellClick(rowIndex)}
                  >
                    {isEditMode && (
                      <td className="px-4 py-3 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveRow(rowIndex);
                          }}
                          disabled={data.rows.length <= 1}
                          className={`p-2 rounded-lg hover:bg-red-50 text-gray-600 hover:text-red-600 transition-colors ${
                            data.rows.length <= 1 ? 'opacity-50 cursor-not-allowed' : ''
                          }`}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    )}
                    {row.map((cell, colIndex) => (
                      <td
                        key={colIndex}
                        className="px-6 py-3 text-gray-700"
                      >
                        {isEditMode ? (
                          <input
                            type="text"
                            value={cell}
                            onChange={(e) => handleTextChange(rowIndex, colIndex, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm hover:border-gray-400 transition-colors"
                            placeholder="Enter value"
                          />
                        ) : (
                          <span className="font-medium text-gray-900">{cell || '-'}</span>
                        )}
                        {isEditMode && mergeMode && isCellSelected(rowIndex) && colIndex === 0 && (
                          <div className="text-xs text-green-600 mt-1 font-semibold">✓ Selected</div>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        
      </div>

      {showAuthModal && (
        <div className="fixed inset-0 bg-gray-900/75 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-2xl p-8 max-w-md w-full mx-4">
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                <Lock className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Authentication Required</h2>
              <p className="text-gray-600">Enter password to enable edit mode</p>
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 mb-6"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleAuthSubmit();
                }
              }}
            />
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowAuthModal(false);
                  setPassword('');
                }}
                className="flex-1 px-4 py-3 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleAuthSubmit}
                className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}