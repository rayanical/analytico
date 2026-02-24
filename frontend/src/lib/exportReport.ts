import jsPDF from 'jspdf';
import { toCanvas } from 'html-to-image';

interface ExportDashboardOptions {
  filename: string;
  datasetLabel: string;
  widgetCount: number;
  targetSelector: string;
  excludeSelectors?: string[];
  useExportModeClass?: boolean;
}

interface PageSlice {
  startCssPx: number;
  endCssPx: number;
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

export async function exportDashboardReport(options: ExportDashboardOptions) {
  const {
    filename,
    datasetLabel,
    widgetCount,
    targetSelector,
    excludeSelectors = [],
    useExportModeClass = true,
  } = options;

  const root = document.querySelector<HTMLElement>(targetSelector);
  if (!root) {
    throw new Error(`Dashboard export target not found: ${targetSelector}`);
  }

  const hiddenNodes: Array<{ node: HTMLElement; previousDisplay: string }> = [];

  try {
    if (useExportModeClass) {
      document.body.classList.add('is-exporting');
    }

    excludeSelectors.forEach(selector => {
      document.querySelectorAll<HTMLElement>(selector).forEach(node => {
        hiddenNodes.push({ node, previousDisplay: node.style.display });
        node.style.display = 'none';
      });
    });

    await new Promise(resolve => setTimeout(resolve, 60));

    const rootRect = root.getBoundingClientRect();
    const widgetBottoms = Array.from(root.querySelectorAll<HTMLElement>('.react-grid-item'))
      .map(node => {
        const rect = node.getBoundingClientRect();
        return rect.bottom - rootRect.top;
      })
      .filter(value => Number.isFinite(value) && value > 0)
      .sort((a, b) => a - b);

    const excluded = new Set<HTMLElement>();
    excludeSelectors.forEach(selector => {
      document.querySelectorAll<HTMLElement>(selector).forEach(node => excluded.add(node));
    });
    const canvas = await toCanvas(root, {
      cacheBust: true,
      pixelRatio: 2,
      backgroundColor: '#09090b',
      width: root.scrollWidth,
      height: root.scrollHeight,
      style: {
        width: `${root.scrollWidth}px`,
        height: `${root.scrollHeight}px`,
      },
      filter: (node) => {
        if (!(node instanceof HTMLElement)) return true;
        for (const excludedNode of excluded) {
          if (excludedNode === node || excludedNode.contains(node)) {
            return false;
          }
        }
        return true;
      },
    });

    const pdf = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4', compress: true });
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 20;
    const headerHeight = 40;
    const availableWidth = pageWidth - margin * 2;
    const availableHeight = pageHeight - margin * 2 - headerHeight;

    const ptPerCssPx = availableWidth / root.scrollWidth;
    const maxPageCssHeight = Math.max(1, Math.floor(availableHeight / ptPerCssPx));
    const canvasScale = canvas.height / root.scrollHeight;

    const slices: PageSlice[] = [];
    let cursor = 0;
    while (cursor < root.scrollHeight) {
      const idealEnd = Math.min(cursor + maxPageCssHeight, root.scrollHeight);
      if (idealEnd >= root.scrollHeight) {
        slices.push({ startCssPx: cursor, endCssPx: root.scrollHeight });
        break;
      }

      const minFill = cursor + Math.floor(maxPageCssHeight * 0.55);
      const candidate = widgetBottoms
        .filter(bottom => bottom > minFill && bottom <= idealEnd)
        .pop();

      const pageEnd = candidate ?? idealEnd;
      slices.push({ startCssPx: cursor, endCssPx: pageEnd });
      cursor = pageEnd;
    }

    let page = 1;

    for (const slice of slices) {
      if (page > 1) {
        pdf.addPage('a4', 'landscape');
      }
      renderHeader(pdf, pageWidth, margin, datasetLabel, widgetCount, page);

      const startPx = Math.floor(slice.startCssPx * canvasScale);
      const endPx = Math.ceil(slice.endCssPx * canvasScale);
      const currentSliceHeight = Math.max(1, endPx - startPx);
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
        startPx,
        canvas.width,
        currentSliceHeight,
        0,
        0,
        canvas.width,
        currentSliceHeight,
      );

      const sliceUrl = sliceCanvas.toDataURL('image/png');
      const renderedHeight = (slice.endCssPx - slice.startCssPx) * ptPerCssPx;
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

      page += 1;
    }

    pdf.save(filename);
  } finally {
    hiddenNodes.forEach(({ node, previousDisplay }) => {
      node.style.display = previousDisplay;
    });
    if (useExportModeClass) {
      document.body.classList.remove('is-exporting');
    }
  }
}
