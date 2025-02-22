import React from 'react';
import { HeaderSimple } from '../lib/HeaderSimple';
import { FooterCentered } from '../lib/FooterCentered';
import JupyterNotebookViewer from '../lib/Jupiter'; // Import the JupyterNotebookViewer component

const ApplicationPage: React.FC = () => {
  // Define the URL of the Jupyter notebook
  const notebookUrl = 'https://example.com/my-notebook.ipynb';

  return (
    <div>
      <HeaderSimple />
      <main>
        <JupyterNotebookViewer notebookUrl={notebookUrl} /> {/* Pass the notebookUrl prop */}
      </main>
      <FooterCentered />
    </div>
  );
};

export default ApplicationPage;