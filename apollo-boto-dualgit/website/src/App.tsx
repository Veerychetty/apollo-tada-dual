import { useState, useEffect } from 'react';
import DualTyreInterface from './components/DualTyreInterface';
import RecipesSidebar from './components/RecipesSidebar';
import { Recipe } from './types';

function App() {
  const [showRecipes, setShowRecipes] = useState(false);
  const [recipes, setRecipes] = useState<Recipe[]>([]);

  useEffect(() => {
    const savedRecipes = localStorage.getItem('tyreRecipes');
    if (savedRecipes) {
      setRecipes(JSON.parse(savedRecipes));
    }
  }, []);






  const handleDownloadRecipe = (recipe: Recipe) => {
    const dataStr = JSON.stringify(recipe, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${recipe.name.replace(/\s+/g, '-').toLowerCase()}-${recipe.id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDeleteRecipe = (id: string) => {
    const updatedRecipes = recipes.filter(r => r.id !== id);
    setRecipes(updatedRecipes);
    localStorage.setItem('tyreRecipes', JSON.stringify(updatedRecipes));
  };

  const handleCombinedResults = (data: any) => {
    // Handle combined results if needed
    console.log('Combined results:', data);
  };

  return (
    <div className="min-h-screen bg-white">
      <DualTyreInterface
        onCombinedResults={handleCombinedResults}
      />


      {showRecipes && (
        <RecipesSidebar
          recipes={recipes}
          onClose={() => setShowRecipes(false)}
          onLoad={handleLoadRecipe}
          onDownload={handleDownloadRecipe}
          onDelete={handleDeleteRecipe}
        />
      )}
    </div>
  );
}

export default App;
