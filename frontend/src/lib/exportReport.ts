import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

interface ExportDashboardOptions {
  filename: string;
  datasetLabel: string;
  widgetCount: number;
}

function renderHeader(
  pdf: jsPDF,
  pageWidth: number,
  margin: number,
  datasetLabel: string,
  widgetCount: number,
  pageNumber: number,
) {
  const generatedAt = new Date().toLocaleString();
  pdf.setFontSize(14);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Analytico Dashboard Report', margin, margin + 14);

  pdf.setFontSize(9);
  pdf.setFont('helvetica', 'normal');
  pdf.text(`Dataset: ${datasetLabel}`, margin, margin + 30);
  pdf.text(`Widgets: ${widgetCount}`, margin + 220, margin + 30);
  pdf.text(`Generated: ${generatedAt}`, margin + 320, margin + 30);
  pdf.text(`Page ${pageNumber}`, pageWidth - margin - 40, margin + 30);

  pdf.setDrawColor(180);
  pdf.line(margin, margin + 36, pageWidth - margin, margin + 36);
}

export async function exportDashboardReport(
  element: HTMLElement,
  options: ExportDashboardOptions,
) {
  const canvas = await html2canvas(element, {
    scale: 2,
    useCORS: true,
    backgroundColor: '#09090b',
    windowWidth: element.scrollWidth,
    windowHeight: element.scrollHeight,
  });

  const pdf = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4', compress: true });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 20;
  const headerHeight = 40;
  const availableWidth = pageWidth - margin * 2;
  const availableHeight = pageHeight - margin * 2 - headerHeight;

  const pxToPt = availableWidth / canvas.width;
  const sliceHeightPx = Math.max(1, Math.floor(availableHeight / pxToPt));

  let sourceY = 0;
  let page = 1;

  while (sourceY < canvas.height) {
    if (page > 1) {
      pdf.addPage('a4', 'landscape');
    }
    renderHeader(pdf, pageWidth, margin, options.datasetLabel, options.widgetCount, page);

    const remainingPx = canvas.height - sourceY;
    const currentSliceHeight = Math.min(sliceHeightPx, remainingPx);
    const sliceCanvas = document.createElement('canvas');
    sliceCanvas.width = canvas.width;
    sliceCanvas.height = currentSliceHeight;
    const sliceCtx = sliceCanvas.getContext('2d');
    if (!sliceCtx) {
      throw new Error('Failed to prepare canvas slice for PDF export');
    }

    sliceCtx.drawImage(
      canvas,
      0,
      sourceY,
      canvas.width,
      currentSliceHeight,
      0,
      0,
      canvas.width,
      currentSliceHeight,
    );

    const sliceUrl = sliceCanvas.toDataURL('image/png');
    const renderedHeight = currentSliceHeight * pxToPt;
    pdf.addImage(
      sliceUrl,
      'PNG',
      margin,
      margin + headerHeight,
      availableWidth,
      renderedHeight,
      undefined,
      'FAST',
    );

    sourceY += currentSliceHeight;
    page += 1;
  }

  pdf.save(options.filename);
}
