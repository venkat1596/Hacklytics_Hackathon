import React from 'react';

interface DrawIOEmbedProps {
  diagramUrl?: string; // URL of the diagram to embed (optional)
  width?: string; // Width of the iframe (optional)
  height?: string; // Height of the iframe (optional)
}

const DrawIOEmbed: React.FC<DrawIOEmbedProps> = ({
  diagramUrl,
  width = '100%',
  height = '800px',
}) => {
  // Use provided URL or default Draw.io editor URL
  const drawIoUrl = diagramUrl
    ? `https://viewer.diagrams.net/?lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Mirai.drawio&transparent=1&dark=0#U${encodeURIComponent(
        diagramUrl
      )}`
    : 'https://embed.diagrams.net/?embed=1&ui=atlas&spin=1&proto=json';

  return (
    <iframe
      title="Draw.io Diagram"
      src={drawIoUrl}
      width={width}
      height={height}
      frameBorder="0"
      allowFullScreen
    />
  );
};

export default DrawIOEmbed;
