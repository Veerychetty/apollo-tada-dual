import { X, Clock, Download, FolderOpen, Trash2 } from 'lucide-react';
import { Recipe } from '../types';

interface RecipesSidebarProps {
  recipes: Recipe[];
  onClose: () => void;
  onLoad: (recipe: Recipe) => void;
  onDownload: (recipe: Recipe) => void;
  onDelete: (id: string) => void;
}

export default function RecipesSidebar({
  recipes,
  onClose,
  onLoad,
  onDownload,
  onDelete,
}: RecipesSidebarProps) {
  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="fixed inset-0 z-50 animate-fade-in">
      <div className="absolute inset-0 bg-gray-900/75 backdrop-blur-sm" onClick={onClose}></div>
      <div className="absolute right-0 top-0 bottom-0 w-full max-w-2xl bg-white shadow-2xl animate-slide-in-right overflow-hidden flex flex-col">
        <div className="bg-gray-800 p-8 text-white border-b border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-3xl font-bold">Inspection History</h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-700 transition-colors"
            >
              <X className="w-7 h-7" />
            </button>
          </div>
          <p className="text-gray-300">
            {recipes.length} saved {recipes.length === 1 ? 'session' : 'sessions'}
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-8 bg-gray-50">
          {recipes.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-24 h-24 bg-gray-300 rounded-full flex items-center justify-center mb-6">
                <FolderOpen className="w-12 h-12 text-gray-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800 mb-2">No History Yet</h3>
              <p className="text-gray-600">
                Your saved inspections will appear here
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {recipes.map((recipe) => (
                <div
                  key={recipe.id}
                  className="group relative bg-white rounded-lg border border-gray-200 hover:border-gray-400 p-6 transition-all duration-300 hover:shadow-md"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-gray-900 mb-2">
                        {recipe.name}
                      </h3>
                      <div className="flex items-center gap-2 text-gray-600">
                        <Clock className="w-4 h-4" />
                        <span className="text-sm">{formatDate(recipe.timestamp)}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => onDelete(recipe.id)}
                      className="opacity-0 group-hover:opacity-100 p-2 rounded-lg hover:bg-gray-100 text-gray-600 hover:text-gray-800 transition-all"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="mb-4">
                    <div className="text-sm text-gray-600 mb-2">
                      {recipe.data.rows.length} rows × {recipe.data.headers.length} columns
                    </div>
                    {recipe.images && (
                      <div className="flex gap-2 mb-3">
                        <img
                          src={recipe.images.top}
                          alt="Top view"
                          className="w-20 h-20 object-cover rounded border border-gray-300"
                        />
                        <img
                          src={recipe.images.bottom}
                          alt="Bottom view"
                          className="w-20 h-20 object-cover rounded border border-gray-300"
                        />
                      </div>
                    )}
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => onLoad(recipe)}
                      className="flex-1 px-4 py-2.5 rounded-lg font-semibold bg-gray-800 text-white hover:bg-gray-900 transition-all"
                    >
                      Load Session
                    </button>
                    <button
                      onClick={() => onDownload(recipe)}
                      className="px-4 py-2.5 rounded-lg font-semibold bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
