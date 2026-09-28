import React, { useEffect, useState } from 'react';
import { ConcertView } from './views/ConcertView';
import { TopDownView } from './views/TopDownView';

export const App: React.FC = () => {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Extremely simple routing
  if (currentPath === '/concert') {
    return <ConcertView />;
  }

  // Default route is the simple top-down view
  return <TopDownView />;
};
