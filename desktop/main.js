const { app, BrowserWindow, Tray, Menu, globalShortcut, nativeImage } = require("electron");
const path = require("path");

const APP_URL = "http://localhost:3000";

let mainWindow = null;
let tray = null;
app.isQuitting = false;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 650,
    frame: false,
    show: false, // start minimized to tray
    backgroundColor: "#0a0a0a",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(APP_URL);

  // Closing the window hides it to the tray instead of quitting.
  mainWindow.on("close", (event) => {
    if (!app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
}

function showWindow() {
  if (!mainWindow) return;
  mainWindow.show();
  mainWindow.focus();
}

function toggleWindow() {
  if (!mainWindow) return;
  if (mainWindow.isVisible() && mainWindow.isFocused()) {
    mainWindow.hide();
  } else {
    showWindow();
  }
}

function createTray() {
  const icon = nativeImage.createFromPath(
    path.join(__dirname, "app", "favicon.ico")
  );
  tray = new Tray(icon);
  tray.setToolTip("AIOS");

  const contextMenu = Menu.buildFromTemplate([
    { label: "Show", click: showWindow },
    { type: "separator" },
    {
      label: "Quit",
      click: () => {
        app.isQuitting = true;
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(contextMenu);
  tray.on("double-click", showWindow);
}

app.whenReady().then(() => {
  createWindow();
  createTray();

  globalShortcut.register("CommandOrControl+Shift+Space", toggleWindow);

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Keep running in the tray when all windows are closed.
app.on("window-all-closed", (event) => {
  event.preventDefault();
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});
