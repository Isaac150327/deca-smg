// Paste this file into Extensions > Apps Script from the destination Sheet.
// Replace the token below with a long random value, then run setup() once.
const SETUP_TOKEN = "REPLACE_WITH_A_LONG_RANDOM_TOKEN";

function setup() {
  if (SETUP_TOKEN === "REPLACE_WITH_A_LONG_RANDOM_TOKEN") {
    throw new Error("Replace SETUP_TOKEN before running setup().");
  }
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  PropertiesService.getScriptProperties().setProperties({
    DECA_SMG_SPREADSHEET_ID: spreadsheet.getId(),
    DECA_SMG_TOKEN: SETUP_TOKEN,
  });
}

function jsonResponse(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}

function safeCell(value) {
  if (value === "") return "";
  if (/^-?(?:\d+\.?\d*|\.\d+)$/.test(value)) return Number(value);
  if (/^[=+\-@]/.test(value)) return "'" + value;
  return value;
}

function doPost(event) {
  try {
    const properties = PropertiesService.getScriptProperties();
    const payload = JSON.parse(event.postData.contents);
    const expectedToken = properties.getProperty("DECA_SMG_TOKEN");
    if (!expectedToken || payload.token !== expectedToken) {
      return jsonResponse({ok: false, error: "invalid token"});
    }
    if (!Array.isArray(payload.rows) || payload.rows.length === 0) {
      return jsonResponse({ok: false, error: "no CSV rows received"});
    }

    const spreadsheet = SpreadsheetApp.openById(
      properties.getProperty("DECA_SMG_SPREADSHEET_ID")
    );
    const worksheetName = payload.worksheet || "Shortlist";
    let sheet = spreadsheet.getSheetByName(worksheetName);
    if (!sheet) sheet = spreadsheet.insertSheet(worksheetName);

    const rows = payload.rows.map(row => row.map(safeCell));
    const width = rows[0].length;
    if (!rows.every(row => row.length === width)) {
      return jsonResponse({ok: false, error: "CSV rows have different widths"});
    }

    const currentFilter = sheet.getFilter();
    if (currentFilter) currentFilter.remove();
    sheet.clearContents();
    sheet.getRange(1, 1, rows.length, width).setValues(rows);
    sheet.setFrozenRows(1);
    sheet.getRange(1, 1, 1, width).setFontWeight("bold");
    sheet.autoResizeColumns(1, width);
    if (rows.length > 1) sheet.getRange(1, 1, rows.length, width).createFilter();

    return jsonResponse({
      ok: true,
      worksheet: worksheetName,
      rows: Math.max(0, rows.length - 1),
    });
  } catch (error) {
    return jsonResponse({ok: false, error: String(error)});
  }
}
