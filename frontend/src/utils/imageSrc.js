export function toRenderableImageSrc(rawValue, mimeType = 'image/png') {
  if (rawValue === null || rawValue === undefined) {
    return null;
  }
  const text = String(rawValue).trim().replace(/\s+/g, '');
  if (!text) {
    return null;
  }
  if (text.startsWith('data:image/')) {
    return text;
  }
  return `data:${mimeType};base64,${text}`;
}

