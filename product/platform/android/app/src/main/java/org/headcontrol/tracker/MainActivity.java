package org.headcontrol.tracker;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.View;
import android.view.WindowInsets;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.function.IntConsumer;

public final class MainActivity extends Activity {
    private static final int CAMERA = 1, NOTIFICATIONS = 2;
    private static final int BACKGROUND = 0xff0c1422, CARD = 0xff172439, TEXT = 0xffedf4ff,
            MUTED = 0xffafc1da, BLUE = 0xff235cba, LIGHT_BLUE = 0xff78b8ff, GREEN = 0xff7ce0af, AMBER = 0xffffd08a;
    private static final String DEMO_INITIALIZED = "demo_v1_initialized", RESPONSE = "demo_response";
    private static final String[] PROFILES = {"Desactivada", "Equilibrada", "Firme", "Fluido preciso", "Fluido preciso · consenso"};
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final LinearLayout[] sections = new LinearLayout[3];
    private final Button[] tabs = new Button[3];
    private SharedPreferences settings;
    private ScrollView scroll;
    private int selectedPage;
    private TextView homeState, cameraState, neutralState, controlState, summary, hint, diagnostics, traceStatus;
    private ProgressBar calibration;
    private Button start, control, stop, retry, trace;
    private String launchError = "";
    private TrackingService displayedSession;
    private boolean wasArmed;

    private final Runnable refresh = new Runnable() {
        @Override public void run() {
            render();
            ui.postDelayed(this, 200);
        }
    };

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        settings = getSharedPreferences(EdgeGestures.SETTINGS, MODE_PRIVATE);
        if (!settings.getBoolean(DEMO_INITIALIZED, false)) {
            settings.edit().putInt(StablePointer.PREF, StablePointer.Profile.CONSENSUS.ordinal())
                    .putInt(RESPONSE, 0).putBoolean(DEMO_INITIALIZED, true).apply();
        }

        LinearLayout shell = column();
        shell.setBackgroundColor(BACKGROUND);
        shell.setPadding(dp(16), dp(12), dp(16), dp(8));
        shell.setOnApplyWindowInsetsListener((view, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                var safe = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                view.setPadding(dp(16)+safe.left, dp(12)+safe.top, dp(16)+safe.right, dp(8)+safe.bottom);
            } else {
                view.setPadding(dp(16)+insets.getSystemWindowInsetLeft(), dp(12)+insets.getSystemWindowInsetTop(),
                        dp(16)+insets.getSystemWindowInsetRight(), dp(8)+insets.getSystemWindowInsetBottom());
            }
            return insets;
        });
        shell.addView(label("HeadControl AI", 27, TEXT));
        shell.addView(label("Control sin manos · Android", 14, MUTED));
        LinearLayout nav = new LinearLayout(this);
        String[] names = {"Inicio", "Ajustes", "Diagnóstico"};
        for (int i = 0; i < tabs.length; i++) {
            final int page = i;
            Button tab = button(names[i], () -> showPage(page));
            tab.setTextSize(13);
            tab.setMinimumWidth(0);
            tab.setPadding(dp(2), 0, dp(2), 0);
            nav.addView(tab, new LinearLayout.LayoutParams(0, dp(52), 1));
            tabs[i] = tab;
        }
        shell.addView(nav);
        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        LinearLayout pages = column();
        scroll.addView(pages);
        for (int i = 0; i < sections.length; i++) {
            sections[i] = column();
            pages.addView(sections[i]);
        }
        shell.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(shell);

        LinearLayout stateCard = card(sections[0], "Seguimiento");
        homeState = addLabel(stateCard, "", 22, LIGHT_BLUE);
        cameraState = addLabel(stateCard, "", 15, TEXT);
        neutralState = addLabel(stateCard, "", 15, TEXT);
        controlState = addLabel(stateCard, "", 15, TEXT);
        calibration = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        calibration.setMax(NeutralCalibration.GOAL);
        stateCard.addView(calibration);
        hint = addLabel(stateCard, "", 14, MUTED);
        summary = addLabel(stateCard, "", 14, MUTED);

        LinearLayout actions = card(sections[0], "Controles");
        start = addButton(actions, "Iniciar cámara y calibrar", () -> begin(true));
        start.setBackgroundTintList(ColorStateList.valueOf(BLUE));
        control = addButton(actions, "Activar control", () -> {
            if (TrackingService.instance != null) TrackingService.instance.toggle();
        });
        stop = addButton(actions, "Detener cámara", () -> {
            if (TrackingService.instance != null) TrackingService.instance.stop("activity_stop");
        });
        retry = addButton(actions, "Reintentar neutral", () -> {
            if (TrackingService.instance != null) TrackingService.instance.retryNeutral();
        });
        addButton(actions, "Ajustes de accesibilidad", () -> startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));

        LinearLayout tuning = card(sections[1], "Respuesta del puntero");
        choices(tuning, "Modo", new String[]{"Directo", "Suave"}, settings.getInt(RESPONSE, 0), index -> {
            settings.edit().putInt(RESPONSE, index).apply();
            if (TrackingService.instance != null) TrackingService.instance.setSmooth(index == 1);
        });
        choices(tuning, "Fijación", PROFILES, profile().ordinal(),
                index -> settings.edit().putInt(StablePointer.PREF, index).apply());
        numbers(tuning, "Corte mínimo del filtro adaptativo", OneEuro.CUTOFF, OneEuro.DEFAULT_CUTOFF,
                new int[]{20, 30, 45}, value -> value/10.0 + " Hz");
        numbers(tuning, "Sensibilidad del filtro adaptativo", OneEuro.BETA, OneEuro.DEFAULT_BETA,
                new int[]{12, 22, 35}, value -> Double.toString(value/100.0));

        LinearLayout navigation = card(sections[1], "Permanencia y navegación");
        timing(navigation, "Permanencia para tocar", EdgeGestures.TAP, EdgeGestures.DEFAULT_TAP, new int[]{600, 800, 1000, 1200});
        timing(navigation, "Permanencia inicial en flechas", EdgeGestures.ARROW, EdgeGestures.DEFAULT_ARROW, new int[]{250, 350, 500, 700});
        timing(navigation, "Intervalo de desplazamiento", EdgeGestures.REPEAT, EdgeGestures.DEFAULT_REPEAT, new int[]{300, 550, 800, 1100});
        addLabel(navigation, "Apunta a los bordes para desplazar. Usa Arrastrar, Atrás, Inicio y Pausa por permanencia.", 14, MUTED);

        LinearLayout technical = card(sections[2], "Rendimiento y seguimiento");
        trace = addButton(technical, "Registrar puntero", () -> {
            if (PointerService.instance != null) PointerService.instance.toggleMotionTrace();
        });
        traceStatus = addLabel(technical, "", 14, MUTED);
        diagnostics = addLabel(technical, "", 13, TEXT);
        diagnostics.setTextIsSelectable(true);
        showPage(state == null ? 0 : state.getInt("page", 0));
        render();
    }

    private void render() {
        TrackingService session = TrackingService.instance;
        if (displayedSession != session) {
            displayedSession = session;
            wasArmed = false;
            if (session != null) session.setSmooth(settings.getInt(RESPONSE, 0) == 1);
        }
        NeutralCalibration.Feedback feedback = session == null ? null : session.neutralFeedback();
        boolean ready = feedback != null && feedback.state == NeutralCalibration.State.READY;
        boolean timeout = feedback != null && feedback.state == NeutralCalibration.State.TIMEOUT;
        boolean live = session != null && session.cameraActive();
        boolean armed = session != null && session.armed();
        wasArmed |= armed;
        String tracking = session == null ? TrackingService.status : session.trackingStatus();
        boolean degraded = tracking.contains("degradado") || tracking.contains("antigua") || tracking.contains("Esperando rostro");
        boolean error = timeout || launchError.contains("rechazó") || session == null &&
                (tracking.contains("Error") || tracking.contains("failed") || tracking.contains("timeout") || tracking.contains("denied"));
        boolean paused = ready && (degraded || tracking.contains("pausado") || wasArmed && !armed);
        String state = error ? "Error" : paused ? "Pausado" : armed ? "Activo"
                : session == null ? "Inactivo" : ready ? "Listo" : "Calibrando";
        homeState.setText(state);
        homeState.setTextColor(error || paused || "Calibrando".equals(state) ? AMBER : armed ? GREEN : LIGHT_BLUE);
        cameraState.setText("Cámara · " + (live ? "Activa" : session == null ? "Detenida" : "Iniciando"));
        neutralState.setText("Neutral · " + (ready ? "Listo" : timeout ? "Reintentar" : feedback == null
                ? "Pendiente" : feedback.progress + " / " + NeutralCalibration.GOAL));
        controlState.setText("Control · " + (session != null && session.activating() ? "Preparando"
                : armed && !degraded ? "Activo" : "Detenido"));
        calibration.setVisibility(feedback != null && !ready ? View.VISIBLE : View.GONE);
        calibration.setProgress(feedback == null ? 0 : feedback.progress);
        String message = !launchError.isEmpty() ? launchError.trim()
                : PointerService.instance == null ? "Habilita HeadControl AI en Accesibilidad."
                : feedback != null && !ready ? feedback.message
                : error ? "Consulta Diagnóstico." : paused ? "El puntero está detenido." : "";
        hint.setText(message);
        hint.setVisibility(message.isEmpty() ? View.GONE : View.VISIBLE);
        summary.setText((settings.getInt(RESPONSE, 0) == 0 ? "Directo" : "Suave")
                + " · " + PROFILES[profile().ordinal()] + "\nPermanencia: "
                + settings.getInt(EdgeGestures.TAP, EdgeGestures.DEFAULT_TAP) + " ms");
        start.setEnabled(session == null);
        stop.setEnabled(session != null);
        control.setEnabled(ready);
        control.setText(armed ? "Pausar control" : session != null && session.activating() ? "Cancelar activación" : "Activar control");
        retry.setVisibility(live && !ready ? View.VISIBLE : View.GONE);
        PointerService pointer = PointerService.instance;
        trace.setEnabled(pointer != null);
        trace.setText(pointer != null && pointer.tracingMotion() ? "Guardar registro del puntero" : "Registrar puntero");
        traceStatus.setText(pointer == null ? "" : pointer.motionTraceStatus());
        if (selectedPage == 2) diagnostics.setText(tracking + "\n\n" + (session == null ? TrackingService.stoppedDiagnostics
                : (feedback == null ? "" : feedback.detail + "\n\n") + session.diagnostics()));
    }

    private StablePointer.Profile profile() {
        return StablePointer.Profile.at(settings.getInt(StablePointer.PREF, StablePointer.Profile.CONSENSUS.ordinal()));
    }

    @Override protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state);
        state.putInt("page", selectedPage);
    }

    private void showPage(int page) {
        selectedPage = page;
        for (int i = 0; i < sections.length; i++) {
            boolean selected = i == page;
            sections[i].setVisibility(selected ? View.VISIBLE : View.GONE);
            tabs[i].setSelected(selected);
            tabs[i].setBackgroundTintList(ColorStateList.valueOf(selected ? 0xff1d4490 : CARD));
        }
        scroll.scrollTo(0, 0);
    }

    private int dp(int amount) { return Math.round(amount * getResources().getDisplayMetrics().density); }

    private LinearLayout column() {
        LinearLayout view = new LinearLayout(this);
        view.setOrientation(LinearLayout.VERTICAL);
        return view;
    }

    private TextView label(String text, int size, int color) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setPadding(0, dp(3), 0, dp(3));
        return view;
    }

    private TextView addLabel(LinearLayout parent, String text, int size, int color) {
        TextView view = label(text, size, color);
        parent.addView(view);
        return view;
    }

    private LinearLayout card(LinearLayout parent, String title) {
        LinearLayout view = column();
        view.setPadding(dp(14), dp(11), dp(14), dp(11));
        GradientDrawable background = new GradientDrawable();
        background.setColor(CARD);
        background.setCornerRadius(dp(13));
        background.setStroke(dp(1), 0xff293e5a);
        view.setBackground(background);
        LinearLayout.LayoutParams position = new LinearLayout.LayoutParams(-1, -2);
        position.bottomMargin = dp(10);
        parent.addView(view, position);
        addLabel(view, title, 17, TEXT);
        return view;
    }

    private Button button(String title, Runnable action) {
        Button button = new Button(this);
        button.setText(title);
        button.setAllCaps(false);
        button.setMinHeight(dp(48));
        button.setOnClickListener(view -> action.run());
        return button;
    }

    private Button addButton(LinearLayout parent, String title, Runnable action) {
        Button button = button(title, action);
        parent.addView(button);
        return button;
    }

    private void timing(LinearLayout parent, String title, String key, int fallback, int[] options) {
        numbers(parent, title, key, fallback, options, value -> value + " ms");
    }

    private void numbers(LinearLayout parent, String title, String key, int fallback, int[] options,
                         java.util.function.IntFunction<String> display) {
        String[] entries = new String[options.length];
        int initial = 0;
        int saved = settings.getInt(key, fallback);
        for (int i = 0; i < options.length; i++) {
            entries[i] = display.apply(options[i]);
            if (options[i] == saved) initial = i;
        }
        choices(parent, title, entries, initial, index -> settings.edit().putInt(key, options[index]).apply());
    }

    private void choices(LinearLayout parent, String title, String[] choices, int initial, IntConsumer select) {
        addLabel(parent, title, 14, MUTED);
        Spinner spinner = new Spinner(this);
        spinner.setMinimumHeight(dp(48));
        spinner.setContentDescription(title);
        spinner.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, choices));
        parent.addView(spinner);
        spinner.setSelection(initial);
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) { select.accept(pos); }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    private void begin(boolean requestNotification) {
        launchError = "";
        if (PointerService.instance == null || TrackingService.instance != null) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA);
            return;
        }
        if (requestNotification && Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, NOTIFICATIONS);
            return;
        }
        try {
            startForegroundService(new Intent(this, TrackingService.class));
        } catch (RuntimeException error) {
            android.util.Log.e("HeadControl", "service_start_denied", error);
            launchError = "\nAndroid rechazó el inicio de cámara. Intenta desde esta pantalla.";
        }
    }

    @Override public void onRequestPermissionsResult(int code, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(code, permissions, grants);
        if (grants.length == 0 || grants[0] != PackageManager.PERMISSION_GRANTED) {
            launchError = "\nAutoriza cámara y notificaciones para iniciar el seguimiento global con un control para detenerlo.";
        } else {
            // The explicit tap remains the entry point after Android closes its permission dialog.
            launchError = "\nPermiso concedido. Pulsa Iniciar cámara.";
        }
    }

    @Override protected void onResume() {
        super.onResume();
        ui.post(refresh);
    }
    @Override protected void onPause() {
        ui.removeCallbacks(refresh);
        super.onPause();
    }
}
