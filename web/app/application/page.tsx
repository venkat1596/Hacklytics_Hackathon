import React from 'react';
import { HeaderSimple } from '../lib/HeaderSimple';
import { FooterCentered } from '../lib/FooterCentered';
import { FaqSimple } from '../lib/FaqSimple'; // Ensure this import is correct
import JupyterNotebookViewer from '../lib/Jupiter'; // Import the JupyterNotebookViewer component

const ApplicationPage: React.FC = () => {
  // Define the URL of the Jupyter notebook
  const notebookUrl = 'https://example.com/my-notebook.ipynb';

  return (
    <div>
      <HeaderSimple />
      <main>
        <FaqSimple /> {/* Ensure this is used correctly */}
      </main>
      <FooterCentered />
    </div>
  );
};

export default ApplicationPage;