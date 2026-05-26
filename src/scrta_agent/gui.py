from __future__ import annotations

import contextlib
import json
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .data_import import prepare_inputs
from .deep_dive import DeepDiveSelection
from .llm import LLMClient
from .schemas import WorkflowConfig
from .utils import slugify, utc_timestamp
from .workflow import HypothesisRegenerationRequested, ScRTAWorkflow


SETTINGS_FILE = ".scrta_gui_settings.json"


class QueueWriter:
    def __init__(self, target_queue: queue.Queue) -> None:
        self.target_queue = target_queue

    def write(self, text: str) -> int:
        if text:
            self.target_queue.put(("log", text))
        return len(text)

    def flush(self) -> None:
        return None


class ScRTAgentLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("scRT-agent Launcher")
        self.geometry("1500x900")
        self.minsize(1200, 720)
        self.message_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.llm_test_thread: threading.Thread | None = None
        self.stop_requested = False

        self.mode_var = tk.StringVar(value="scrna_sctcr")
        self.rna_var = tk.StringVar()
        self.tcr_var = tk.StringVar()
        self.brief_var = tk.StringVar()
        self.analysis_name_var = tk.StringVar(value="scrta_case")
        self.output_dir_var = tk.StringVar(value="runs")
        self.model_var = tk.StringVar(value="gpt-5.4")
        self.analysis_loops_var = tk.StringVar(value="6")
        self.repair_attempts_var = tk.StringVar(value="1")
        self.script_timeout_var = tk.StringVar(value="7200")
        self.status_var = tk.StringVar(value="Ready")
        self.last_run_dir_var = tk.StringVar()
        self.command_preview_var = tk.StringVar()
        self.plan_review_status_var = tk.StringVar(value="Waiting for selected-hypothesis plan")
        self.hypothesis_status_var = tk.StringVar(value="Waiting for generated candidates")
        self.hypothesis_title_var = tk.StringVar()
        self.plan_review_holder: dict[str, object] | None = None
        self.plan_review_event: threading.Event | None = None
        self.current_hypothesis_candidates: dict[str, dict[str, object]] = {}
        self.hypothesis_selection_holder: dict[str, object] | None = None
        self.hypothesis_selection_event: threading.Event | None = None

        self.execute_var = tk.BooleanVar(value=True)
        self.input_prep_llm_var = tk.BooleanVar(value=True)
        self.interactive_plan_review_var = tk.BooleanVar(value=True)
        self.interactive_selection_var = tk.BooleanVar(value=True)
        self.deep_dive_var = tk.BooleanVar(value=True)
        self.mechanism_var = tk.BooleanVar(value=True)
        self.downstream_var = tk.BooleanVar(value=True)

        self._build_layout()
        self._load_settings(silent=True)
        self._bind_preview_updates()
        self._update_command_preview()
        self.after(100, self._poll_queue)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(1, weight=1)
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)
        right.rowconfigure(3, weight=1)

        config = ttk.LabelFrame(left, text="Run Configuration", padding=8)
        config.grid(row=0, column=0, sticky="nsew")
        config.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(config, text="Mode").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            config,
            textvariable=self.mode_var,
            values=["scrna_sctcr"],
            state="readonly",
            width=24,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(config, text="RNA folder(s) or files").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.rna_var, width=64).grid(row=row, column=1, sticky="ew", pady=4)
        rna_buttons = ttk.Frame(config)
        rna_buttons.grid(row=row, column=2, padx=6)
        ttk.Button(rna_buttons, text="Browse Folder", command=lambda: self._browse_folder(self.rna_var)).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(rna_buttons, text="Add Files", command=lambda: self._add_input_files(self.rna_var)).grid(row=0, column=1)
        row += 1

        ttk.Label(config, text="TCR folder(s) or files").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.tcr_var, width=64).grid(row=row, column=1, sticky="ew", pady=4)
        tcr_buttons = ttk.Frame(config)
        tcr_buttons.grid(row=row, column=2, padx=6)
        ttk.Button(tcr_buttons, text="Browse Folder", command=lambda: self._browse_folder(self.tcr_var)).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(tcr_buttons, text="Add Files", command=lambda: self._add_input_files(self.tcr_var)).grid(row=0, column=1)
        row += 1

        ttk.Label(config, text="Research brief").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.brief_var, width=64).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(config, text="Browse", command=self._browse_brief_file).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(config, text="Literature paths").grid(row=row, column=0, sticky="nw", pady=4)
        literature_frame = ttk.Frame(config)
        literature_frame.grid(row=row, column=1, sticky="nsew", pady=4)
        literature_frame.columnconfigure(0, weight=1)
        self.literature_list = tk.Listbox(literature_frame, height=5, exportselection=False)
        self.literature_list.grid(row=0, column=0, sticky="nsew")
        lit_scroll = ttk.Scrollbar(literature_frame, orient="vertical", command=self.literature_list.yview)
        lit_scroll.grid(row=0, column=1, sticky="ns")
        self.literature_list.configure(yscrollcommand=lit_scroll.set)
        literature_buttons = ttk.Frame(config)
        literature_buttons.grid(row=row, column=2, sticky="n", padx=6, pady=4)
        ttk.Button(literature_buttons, text="Add", command=self._add_literature_path).grid(row=0, column=0, pady=2)
        ttk.Button(literature_buttons, text="Remove", command=self._remove_literature_path).grid(row=1, column=0, pady=2)
        row += 1

        ttk.Label(config, text="Analysis name").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.analysis_name_var).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        row += 1

        ttk.Label(config, text="Output dir").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.output_dir_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(config, text="Browse", command=self._browse_output_dir).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(config, text="Model").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.model_var).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        row += 1

        ttk.Label(config, text="Analysis loops").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.analysis_loops_var, width=12).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        ttk.Label(config, text="Repair attempts").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.repair_attempts_var, width=12).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        ttk.Label(config, text="Script timeout").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.script_timeout_var, width=12).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        loops = ttk.LabelFrame(left, text="Execution Options", padding=8)
        loops.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for idx in range(2):
            loops.columnconfigure(idx, weight=1)
        ttk.Checkbutton(loops, text="Prepare inputs with LLM", variable=self.input_prep_llm_var).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Checkbutton(loops, text="Execute scripts", variable=self.execute_var).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(loops, text="Interactive plan review", variable=self.interactive_plan_review_var).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Checkbutton(loops, text="Interactive hypothesis selection", variable=self.interactive_selection_var).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Checkbutton(loops, text="Deep-dive loop", variable=self.deep_dive_var).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Checkbutton(loops, text="Mechanism loop", variable=self.mechanism_var).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Checkbutton(loops, text="Downstream loop", variable=self.downstream_var).grid(row=3, column=0, sticky="w", pady=4)

        actions = ttk.LabelFrame(left, text="Actions", padding=8)
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(actions, text="Start Run", command=self._start_run)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=4)
        self.stop_button = ttk.Button(actions, text="Stop Run", command=self._request_stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=4)
        ttk.Button(actions, text="Save Current Settings", command=self._save_settings).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(actions, text="Reload Settings", command=lambda: self._load_settings(silent=False)).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=4)
        ttk.Button(actions, text="Test LLM Connection", command=self._test_llm_connection).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=4,
        )

        status = ttk.LabelFrame(right, text="Run Status", padding=8)
        status.grid(row=0, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)
        ttk.Label(status, text="Status").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(status, text="Last run dir").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(status, textvariable=self.last_run_dir_var, state="readonly").grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(status, text="Command preview").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(status, textvariable=self.command_preview_var, state="readonly").grid(row=2, column=1, sticky="ew", pady=4)

        hypothesis = ttk.LabelFrame(right, text="Hypothesis Review", padding=8)
        hypothesis.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        hypothesis.columnconfigure(1, weight=1)
        ttk.Label(hypothesis, text="Status").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(hypothesis, textvariable=self.hypothesis_status_var).grid(row=0, column=1, sticky="w", pady=4)

        candidate_frame = ttk.Frame(hypothesis)
        candidate_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        candidate_frame.columnconfigure(0, weight=1)
        self.hypothesis_list = tk.Listbox(candidate_frame, height=4, exportselection=False)
        self.hypothesis_list.grid(row=0, column=0, sticky="ew")
        self.hypothesis_list.bind("<<ListboxSelect>>", self._load_hypothesis_review_candidate)
        hyp_scroll = ttk.Scrollbar(candidate_frame, orient="vertical", command=self.hypothesis_list.yview)
        hyp_scroll.grid(row=0, column=1, sticky="ns")
        self.hypothesis_list.configure(yscrollcommand=hyp_scroll.set)

        ttk.Label(hypothesis, text="Title").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(hypothesis, textvariable=self.hypothesis_title_var).grid(row=2, column=1, sticky="ew", pady=4)

        notebook = ttk.Notebook(hypothesis)
        notebook.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
        self.hypothesis_statement_text = self._add_text_tab(notebook, "Statement")
        self.hypothesis_explanation_text = self._add_text_tab(notebook, "Explanation")
        self.hypothesis_tests_text = self._add_text_tab(notebook, "Required Tests")
        self.hypothesis_falsify_text = self._add_text_tab(notebook, "Falsification")
        self.hypothesis_sources_text = self._add_text_tab(notebook, "Source Tables", height=4)

        hypothesis_buttons = ttk.Frame(hypothesis)
        hypothesis_buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        hypothesis_buttons.columnconfigure(0, weight=1)
        hypothesis_buttons.columnconfigure(1, weight=1)
        hypothesis_buttons.columnconfigure(2, weight=1)
        self.confirm_hypothesis_button = ttk.Button(
            hypothesis_buttons,
            text="Use Selected Hypothesis and Continue",
            command=self._confirm_hypothesis_review,
            state="disabled",
        )
        self.confirm_hypothesis_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.regenerate_hypothesis_button = ttk.Button(
            hypothesis_buttons,
            text="Regenerate Hypotheses",
            command=self._regenerate_hypotheses,
            state="disabled",
        )
        self.regenerate_hypothesis_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.cancel_hypothesis_button = ttk.Button(
            hypothesis_buttons,
            text="Cancel Hypothesis Selection",
            command=self._cancel_hypothesis_review,
            state="disabled",
        )
        self.cancel_hypothesis_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        plan_review = ttk.LabelFrame(right, text="Plan Review / User Feedback", padding=8)
        plan_review.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        plan_review.columnconfigure(0, weight=1)
        ttk.Label(plan_review, textvariable=self.plan_review_status_var).grid(row=0, column=0, sticky="w", pady=4)

        plan_notebook = ttk.Notebook(plan_review)
        plan_notebook.grid(row=1, column=0, sticky="ew", pady=4)
        self.plan_review_context_text = self._add_text_tab(plan_notebook, "Next Analyses", height=7)
        self.plan_review_feedback_text = self._add_text_tab(plan_notebook, "Your Changes", height=5)

        plan_buttons = ttk.Frame(plan_review)
        plan_buttons.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        plan_buttons.columnconfigure(0, weight=1)
        plan_buttons.columnconfigure(1, weight=1)
        self.confirm_plan_button = ttk.Button(
            plan_buttons,
            text="Approve Plan and Continue",
            command=self._confirm_plan_review,
            state="disabled",
        )
        self.confirm_plan_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.cancel_plan_button = ttk.Button(
            plan_buttons,
            text="Cancel Plan Review",
            command=self._cancel_plan_review,
            state="disabled",
        )
        self.cancel_plan_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        log_frame = ttk.LabelFrame(right, text="Execution Log", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _bind_preview_updates(self) -> None:
        variables = [
            self.rna_var,
            self.tcr_var,
            self.brief_var,
            self.analysis_name_var,
            self.output_dir_var,
            self.model_var,
            self.analysis_loops_var,
            self.repair_attempts_var,
            self.script_timeout_var,
        ]
        for variable in variables:
            variable.trace_add("write", lambda *_: self._update_command_preview())

    def _add_text_tab(self, notebook: ttk.Notebook, label: str, height: int = 6) -> tk.Text:
        frame = ttk.Frame(notebook, padding=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text = tk.Text(frame, height=height, wrap="word")
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        notebook.add(frame, text=label)
        return text

    def _browse_input(self, target_var: tk.StringVar) -> None:
        paths = filedialog.askopenfilenames(title="Select input files")
        if paths:
            target_var.set(";".join(paths))
            return
        directory = filedialog.askdirectory(title="Select input directory")
        if directory:
            target_var.set(directory)

    def _browse_folder(self, target_var: tk.StringVar) -> None:
        directory = filedialog.askdirectory(title="Select project or sample folder")
        if directory:
            self._append_paths_to_var(target_var, [directory])

    def _add_input_files(self, target_var: tk.StringVar) -> None:
        paths = filedialog.askopenfilenames(
            title="Select input files or archives",
            filetypes=[
                ("Supported files", "*.h5ad *.h5 *.hdf5 *.csv *.tsv *.txt *.loom *.zarr *.zip *.tar *.gz *.tgz *.bz2 *.xz"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._append_paths_to_var(target_var, list(paths))

    @staticmethod
    def _append_paths_to_var(target_var: tk.StringVar, paths: list[str]) -> None:
        existing = [part.strip() for part in target_var.get().split(";") if part.strip()]
        for path in paths:
            if path and path not in existing:
                existing.append(path)
        target_var.set(";".join(existing))

    def _browse_brief_file(self) -> None:
        path = filedialog.askopenfilename(title="Select research brief file")
        if path:
            self.brief_var.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_var.set(path)

    def _add_literature_path(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select literature cards or RAG index",
            filetypes=[
                ("Supported files", "*.csv *.jsonl *.json *.md *.txt"),
                ("All files", "*.*"),
            ],
        )
        for path in paths:
            self.literature_list.insert(tk.END, path)
        self._update_command_preview()

    def _remove_literature_path(self) -> None:
        for index in reversed(self.literature_list.curselection()):
            self.literature_list.delete(index)
        self._update_command_preview()

    def _start_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Run in progress", "A run is already in progress.")
            return
        if not self.rna_var.get().strip() or not self.tcr_var.get().strip():
            messagebox.showerror("Missing inputs", "RNA input(s) and TCR input(s) are required.")
            return
        if self.interactive_selection_var.get() and (not self.execute_var.get() or not self.deep_dive_var.get()):
            messagebox.showerror(
                "Hypothesis review unavailable",
                "Interactive hypothesis selection requires Execute scripts and Deep-dive loop to be enabled.",
            )
            return
        self.stop_requested = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Running")
        self._reset_plan_review("Waiting for selected-hypothesis plan")
        self._reset_hypothesis_review("Waiting for generated candidates")
        self.log_text.delete("1.0", tk.END)
        self._update_command_preview()
        settings = self._collect_settings()
        self.worker_thread = threading.Thread(target=self._run_worker, args=(settings,), daemon=True)
        self.worker_thread.start()

    def _test_llm_connection(self) -> None:
        if self.llm_test_thread and self.llm_test_thread.is_alive():
            messagebox.showinfo("LLM test in progress", "An LLM connection test is already running.")
            return
        model = self.model_var.get().strip() or "gpt-5.4"
        self.status_var.set("Testing LLM")
        self._append_log(f"\nTesting LLM model access: {model}\n")
        self.llm_test_thread = threading.Thread(target=self._run_llm_test_worker, args=(model,), daemon=True)
        self.llm_test_thread.start()

    def _run_llm_test_worker(self, model: str) -> None:
        try:
            client = LLMClient(model=model, use_llm=True)
            client.require_ready()
            self.message_queue.put(("llm_test_ok", model))
        except Exception as exc:
            self.message_queue.put(("llm_test_failed", model, str(exc)))

    def _request_stop(self) -> None:
        self.stop_requested = True
        self._append_log("\nStop requested. The current LLM or script call will finish before the workflow can stop.\n")
        self.status_var.set("Stop requested")

    def _run_worker(self, settings: dict) -> None:
        writer = QueueWriter(self.message_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                print("Starting scRT-agent GUI run.")
                llm = LLMClient(model=settings["model"], use_llm=True)
                print(f"Testing LLM model access before data preparation: {settings['model']}")
                llm.require_ready()
                prepared_dir = (
                    Path(settings["output_dir"])
                    / "prepared_inputs"
                    / f"{slugify(settings['analysis_name'], 'scrta_case')}_{utc_timestamp()}"
                )
                print("Preparing input files...")
                prepared = prepare_inputs(
                    rna_inputs=settings["rna_inputs"],
                    tcr_inputs=settings["tcr_inputs"],
                    output_dir=prepared_dir,
                    llm=llm,
                    analysis_name=settings["analysis_name"],
                    require_llm_plan=bool(settings["input_prep_llm"]),
                )
                print(f"Prepared RNA h5ad: {prepared.rna_h5ad_path}")
                print(f"Prepared TCR table: {prepared.tcr_path}")
                if self.stop_requested:
                    raise RuntimeError("Run stopped before workflow execution.")

                brief_value = settings["research_brief"]
                brief_path = brief_value if brief_value and Path(brief_value).exists() else None
                brief_text = "" if brief_path else brief_value
                literature_cards, rag_index = self._resolve_literature_paths(settings["literature_paths"])
                config = WorkflowConfig(
                    rna_h5ad_path=prepared.rna_h5ad_path,
                    tcr_path=prepared.tcr_path,
                    analysis_name=settings["analysis_name"],
                    output_root=settings["output_dir"],
                    research_brief_path=brief_path,
                    research_brief=brief_text,
                    literature_cards_path=literature_cards,
                    rag_index_path=rag_index,
                    execute_script=bool(settings["execute"]),
                    use_llm=True,
                    model=settings["model"],
                    analysis_loops=int(settings["analysis_loops"]),
                    repair_attempts=int(settings["repair_attempts"]),
                    script_timeout_seconds=int(settings["script_timeout"]),
                    deep_dive_enabled=bool(settings["deep_dive"]),
                    mechanism_loop_enabled=bool(settings["mechanism"]),
                    downstream_analysis_enabled=bool(settings["downstream"]),
                    interactive_hypothesis_selection=bool(settings["interactive_selection"]),
                    interactive_plan_review=bool(settings["interactive_plan_review"]),
                )
                workflow = ScRTAWorkflow(
                    config,
                    llm=llm,
                    hypothesis_selection_callback=self._request_hypothesis_selection,
                    plan_review_callback=self._request_plan_review,
                )
                state = workflow.run()
                self.message_queue.put(("run_complete", str(state.run_dir)))
        except RuntimeError as exc:
            self.message_queue.put(("run_error", f"Run failed before completion.\n\n{exc}\n"))
        except Exception:
            self.message_queue.put(("run_error", traceback.format_exc()))

    def _request_plan_review(self, plan_context: str) -> str:
        holder: dict[str, object] = {}
        event = threading.Event()
        self.message_queue.put(("plan_review", plan_context, holder, event))
        event.wait()
        error = holder.get("error")
        if error:
            raise RuntimeError(str(error))
        return str(holder.get("feedback", ""))

    def _request_hypothesis_selection(self, candidates: dict[str, dict[str, object]]) -> DeepDiveSelection:
        holder: dict[str, object] = {}
        event = threading.Event()
        self.message_queue.put(("hypothesis_review", candidates, holder, event))
        event.wait()
        error = holder.get("error")
        if error:
            raise RuntimeError(str(error))
        if holder.get("regenerate"):
            raise HypothesisRegenerationRequested("User requested regenerated hypothesis candidates from the GUI.")
        selection = holder.get("selection")
        if not isinstance(selection, DeepDiveSelection):
            raise RuntimeError("No hypothesis was selected.")
        return selection

    def _open_hypothesis_dialog(
        self,
        candidates: dict[str, dict[str, object]],
        holder: dict[str, object],
        event: threading.Event,
    ) -> None:
        dialog = HypothesisSelectionDialog(self, candidates)
        self.wait_window(dialog)
        if dialog.selection is None:
            holder["error"] = "Hypothesis selection was cancelled."
        else:
            holder["selection"] = dialog.selection
        event.set()

    def _populate_plan_review(
        self,
        plan_context: str,
        holder: dict[str, object],
        event: threading.Event,
    ) -> None:
        self.plan_review_holder = holder
        self.plan_review_event = event
        self._set_text(self.plan_review_context_text, plan_context)
        self._set_text(self.plan_review_feedback_text, "")
        self.plan_review_status_var.set("Review the selected-hypothesis plan, add changes, then continue")
        self.confirm_plan_button.configure(state="normal")
        self.cancel_plan_button.configure(state="normal")
        self._append_log(
            "\nA selected-hypothesis plan is ready. Add requested plan changes in the Plan Review panel, "
            "then click Approve Plan and Continue.\n"
        )

    def _confirm_plan_review(self) -> None:
        if self.plan_review_holder is None or self.plan_review_event is None:
            messagebox.showinfo("No pending plan review", "No plan review is currently pending.")
            return
        feedback = self._get_text(self.plan_review_feedback_text)
        self.plan_review_holder["feedback"] = feedback
        self.plan_review_event.set()
        self.plan_review_holder = None
        self.plan_review_event = None
        self.confirm_plan_button.configure(state="disabled")
        self.cancel_plan_button.configure(state="disabled")
        self.plan_review_status_var.set("Plan review confirmed; workflow is continuing")
        self.status_var.set("Running")
        if feedback:
            self._append_log("\nPlan feedback submitted. The planning agent will revise the plan and script before execution.\n")
        else:
            self._append_log("\nPlan approved without additional changes.\n")

    def _cancel_plan_review(self) -> None:
        if self.plan_review_holder is None or self.plan_review_event is None:
            return
        self.plan_review_holder["error"] = "Plan review was cancelled."
        self.plan_review_event.set()
        self.plan_review_holder = None
        self.plan_review_event = None
        self.confirm_plan_button.configure(state="disabled")
        self.cancel_plan_button.configure(state="disabled")
        self.plan_review_status_var.set("Plan review cancelled")
        self.status_var.set("Stopping")

    def _reset_plan_review(self, status: str) -> None:
        self.plan_review_holder = None
        self.plan_review_event = None
        self._set_text(self.plan_review_context_text, "")
        self._set_text(self.plan_review_feedback_text, "")
        self.confirm_plan_button.configure(state="disabled")
        self.cancel_plan_button.configure(state="disabled")
        self.plan_review_status_var.set(status)

    def _populate_hypothesis_review(
        self,
        candidates: dict[str, dict[str, object]],
        holder: dict[str, object],
        event: threading.Event,
    ) -> None:
        self.current_hypothesis_candidates = candidates
        self.hypothesis_selection_holder = holder
        self.hypothesis_selection_event = event
        self.hypothesis_list.delete(0, tk.END)
        for hyp_id, candidate in sorted(candidates.items()):
            self.hypothesis_list.insert(tk.END, f"{hyp_id}: {candidate.get('title', '')}")
        self.hypothesis_status_var.set("Select, edit, and confirm one hypothesis")
        self.confirm_hypothesis_button.configure(state="normal")
        self.regenerate_hypothesis_button.configure(state="normal")
        self.cancel_hypothesis_button.configure(state="normal")
        if self.hypothesis_list.size():
            self.hypothesis_list.selection_set(0)
            self.hypothesis_list.activate(0)
            self.hypothesis_list.see(0)
            self._load_hypothesis_review_candidate()
        self._append_log(
            "\nHypothesis candidates are ready. Select and edit one in the Hypothesis Review panel, "
            "then click Use Selected Hypothesis and Continue. Click Regenerate Hypotheses to ask "
            "the LLM for a fresh candidate set.\n"
        )

    def _load_hypothesis_review_candidate(self, event: object | None = None) -> None:
        selection = self.hypothesis_list.curselection()
        if not selection:
            return
        label = self.hypothesis_list.get(selection[0])
        hyp_id = label.split(":", 1)[0].strip()
        candidate = self.current_hypothesis_candidates.get(hyp_id)
        if not candidate:
            return
        raw_candidate = candidate.get("raw") if isinstance(candidate.get("raw"), dict) else {}
        self.hypothesis_title_var.set(str(candidate.get("title") or hyp_id))
        self._set_text(self.hypothesis_statement_text, str(candidate.get("hypothesis_statement") or ""))
        self._set_text(self.hypothesis_explanation_text, str(candidate.get("plain_language_explanation") or ""))
        self._set_text(self.hypothesis_tests_text, _list_to_text(raw_candidate.get("key_validation")))
        self._set_text(self.hypothesis_falsify_text, _list_to_text(raw_candidate.get("falsification_criteria")))
        self._set_text(self.hypothesis_sources_text, _list_to_text(raw_candidate.get("required_output_tables")))

    def _confirm_hypothesis_review(self) -> None:
        if self.hypothesis_selection_holder is None or self.hypothesis_selection_event is None:
            messagebox.showinfo("No pending selection", "No hypothesis selection is currently pending.")
            return
        selection = self.hypothesis_list.curselection()
        if not selection and self.hypothesis_list.size() == 1:
            self.hypothesis_list.selection_set(0)
            self.hypothesis_list.activate(0)
            selection = self.hypothesis_list.curselection()
        if not selection:
            messagebox.showerror("No hypothesis selected", "Select a hypothesis before continuing.")
            return
        label = self.hypothesis_list.get(selection[0])
        hyp_id = label.split(":", 1)[0].strip()
        candidate = self.current_hypothesis_candidates.get(hyp_id)
        if not candidate:
            messagebox.showerror("Invalid selection", "The selected hypothesis is no longer available.")
            return
        selected = DeepDiveSelection(
            hypothesis_id=hyp_id,
            title=self.hypothesis_title_var.get().strip() or hyp_id,
            selected_hypothesis=self._get_text(self.hypothesis_statement_text),
            plain_language_explanation=self._get_text(self.hypothesis_explanation_text),
            rationale="Selected and edited through the GUI hypothesis review panel.",
            required_tests=_text_to_list(self._get_text(self.hypothesis_tests_text)),
            falsification_criteria=_text_to_list(self._get_text(self.hypothesis_falsify_text)),
            source_tables=_text_to_list(self._get_text(self.hypothesis_sources_text))
            or ["rag_grounded_hypothesis_candidates.md"],
            selected_candidate_source="gui_hypothesis_review_panel",
            selected_candidate_text=str(candidate.get("source_text") or "").strip(),
            selection_mode="gui_review_panel_candidate_selection_for_deep_dive",
            data_support_level="not_assessed",
        )
        self.hypothesis_selection_holder["selection"] = selected
        self.hypothesis_selection_event.set()
        self.hypothesis_selection_holder = None
        self.hypothesis_selection_event = None
        self.confirm_hypothesis_button.configure(state="disabled")
        self.regenerate_hypothesis_button.configure(state="disabled")
        self.cancel_hypothesis_button.configure(state="disabled")
        self.hypothesis_status_var.set(f"Confirmed {hyp_id}; workflow is continuing")
        self.status_var.set("Running")
        self._append_log(f"\nConfirmed selected hypothesis: {hyp_id}\n")

    def _regenerate_hypotheses(self) -> None:
        if self.hypothesis_selection_holder is None or self.hypothesis_selection_event is None:
            messagebox.showinfo("No pending selection", "No hypothesis selection is currently pending.")
            return
        self.hypothesis_selection_holder["regenerate"] = True
        self.hypothesis_selection_event.set()
        self.hypothesis_selection_holder = None
        self.hypothesis_selection_event = None
        self.confirm_hypothesis_button.configure(state="disabled")
        self.regenerate_hypothesis_button.configure(state="disabled")
        self.cancel_hypothesis_button.configure(state="disabled")
        self.hypothesis_status_var.set("Regenerating hypotheses")
        self.status_var.set("Regenerating hypotheses")
        self._append_log("\nRegenerating hypothesis candidates with the LLM.\n")

    def _cancel_hypothesis_review(self) -> None:
        if self.hypothesis_selection_holder is None or self.hypothesis_selection_event is None:
            return
        self.hypothesis_selection_holder["error"] = "Hypothesis selection was cancelled."
        self.hypothesis_selection_event.set()
        self.hypothesis_selection_holder = None
        self.hypothesis_selection_event = None
        self.confirm_hypothesis_button.configure(state="disabled")
        self.regenerate_hypothesis_button.configure(state="disabled")
        self.cancel_hypothesis_button.configure(state="disabled")
        self.hypothesis_status_var.set("Selection cancelled")
        self.status_var.set("Stopping")

    def _reset_hypothesis_review(self, status: str) -> None:
        self.current_hypothesis_candidates = {}
        self.hypothesis_selection_holder = None
        self.hypothesis_selection_event = None
        self.hypothesis_list.delete(0, tk.END)
        self.hypothesis_title_var.set("")
        for widget in [
            self.hypothesis_statement_text,
            self.hypothesis_explanation_text,
            self.hypothesis_tests_text,
            self.hypothesis_falsify_text,
            self.hypothesis_sources_text,
        ]:
            self._set_text(widget, "")
        self.confirm_hypothesis_button.configure(state="disabled")
        self.regenerate_hypothesis_button.configure(state="disabled")
        self.cancel_hypothesis_button.configure(state="disabled")
        self.hypothesis_status_var.set(status)

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)

    @staticmethod
    def _get_text(widget: tk.Text) -> str:
        return widget.get("1.0", tk.END).strip()

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self.message_queue.get_nowait()
                kind = message[0]
                if kind == "log":
                    self._append_log(message[1])
                elif kind == "hypothesis_dialog":
                    _, candidates, holder, event = message
                    self.status_var.set("Waiting for hypothesis selection")
                    self._open_hypothesis_dialog(candidates, holder, event)
                    self.status_var.set("Running")
                elif kind == "hypothesis_review":
                    _, candidates, holder, event = message
                    self.status_var.set("Waiting for hypothesis selection")
                    self._populate_hypothesis_review(candidates, holder, event)
                elif kind == "plan_review":
                    _, plan_context, holder, event = message
                    self.status_var.set("Waiting for plan review")
                    self._populate_plan_review(plan_context, holder, event)
                elif kind == "run_complete":
                    self.status_var.set("Finished")
                    self.last_run_dir_var.set(message[1])
                    self._append_log(f"\nRun directory: {message[1]}\n")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                elif kind == "run_error":
                    self.status_var.set("Error")
                    self._append_log("\n" + message[1] + "\n")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                elif kind == "llm_test_ok":
                    self.status_var.set("Ready")
                    self._append_log(f"LLM connection test passed for model: {message[1]}\n")
                    messagebox.showinfo("LLM test passed", f"Model is accessible: {message[1]}")
                elif kind == "llm_test_failed":
                    self.status_var.set("Ready")
                    self._append_log(
                        f"LLM connection test failed for model: {message[1]}\n{message[2]}\n"
                    )
                    messagebox.showerror(
                        "LLM test failed",
                        (
                            f"The configured token cannot use model `{message[1]}` or the endpoint is unavailable.\n\n"
                            "Choose a model that this API key can access, then test again."
                        ),
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _collect_settings(self) -> dict:
        return {
            "mode": self.mode_var.get(),
            "rna_inputs": self.rna_var.get().strip(),
            "tcr_inputs": self.tcr_var.get().strip(),
            "research_brief": self.brief_var.get().strip(),
            "literature_paths": list(self.literature_list.get(0, tk.END)),
            "analysis_name": self.analysis_name_var.get().strip() or "scrta_case",
            "output_dir": self.output_dir_var.get().strip() or "runs",
            "model": self.model_var.get().strip() or "gpt-5.4",
            "analysis_loops": self.analysis_loops_var.get().strip() or "6",
            "repair_attempts": self.repair_attempts_var.get().strip() or "1",
            "script_timeout": self.script_timeout_var.get().strip() or "7200",
            "execute": self.execute_var.get(),
            "input_prep_llm": self.input_prep_llm_var.get(),
            "interactive_plan_review": self.interactive_plan_review_var.get(),
            "interactive_selection": self.interactive_selection_var.get(),
            "deep_dive": self.deep_dive_var.get(),
            "mechanism": self.mechanism_var.get(),
            "downstream": self.downstream_var.get(),
        }

    def _apply_settings(self, settings: dict) -> None:
        self.mode_var.set(settings.get("mode", "scrna_sctcr"))
        self.rna_var.set(settings.get("rna_inputs", ""))
        self.tcr_var.set(settings.get("tcr_inputs", ""))
        self.brief_var.set(settings.get("research_brief", ""))
        self.analysis_name_var.set(settings.get("analysis_name", "scrta_case"))
        self.output_dir_var.set(settings.get("output_dir", "runs"))
        self.model_var.set(settings.get("model", "gpt-5.4"))
        self.analysis_loops_var.set(str(settings.get("analysis_loops", "6")))
        self.repair_attempts_var.set(str(settings.get("repair_attempts", "1")))
        self.script_timeout_var.set(str(settings.get("script_timeout", "7200")))
        self.execute_var.set(bool(settings.get("execute", True)))
        self.input_prep_llm_var.set(bool(settings.get("input_prep_llm", True)))
        self.interactive_plan_review_var.set(bool(settings.get("interactive_plan_review", True)))
        self.interactive_selection_var.set(bool(settings.get("interactive_selection", True)))
        self.deep_dive_var.set(bool(settings.get("deep_dive", True)))
        self.mechanism_var.set(bool(settings.get("mechanism", True)))
        self.downstream_var.set(bool(settings.get("downstream", True)))
        self.literature_list.delete(0, tk.END)
        for path in settings.get("literature_paths", []):
            self.literature_list.insert(tk.END, path)

    def _save_settings(self) -> None:
        path = Path.cwd() / SETTINGS_FILE
        path.write_text(json.dumps(self._collect_settings(), indent=2), encoding="utf-8")
        messagebox.showinfo("Settings saved", f"Settings saved to {path}")

    def _load_settings(self, silent: bool) -> None:
        path = Path.cwd() / SETTINGS_FILE
        if not path.exists():
            if not silent:
                messagebox.showinfo("No settings", f"No settings file found at {path}")
            return
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
            self._apply_settings(settings)
            self._update_command_preview()
            if not silent:
                messagebox.showinfo("Settings loaded", f"Settings loaded from {path}")
        except Exception as exc:
            if not silent:
                messagebox.showerror("Settings error", str(exc))

    def _update_command_preview(self) -> None:
        command = [
            "scrta-agent",
            "run",
            "--rna",
            "<prepared_rna.h5ad>",
            "--tcr",
            "<prepared_tcr.csv>",
            "--analysis-name",
            self.analysis_name_var.get() or "scrta_case",
            "--out",
            self.output_dir_var.get() or "runs",
            "--model",
            self.model_var.get() or "gpt-5.4",
        ]
        if self.execute_var.get():
            command.append("--execute")
        if self.interactive_plan_review_var.get():
            command.append("--interactive-plan-review")
        if self.interactive_selection_var.get():
            command.append("--interactive-hypothesis-selection")
        self.command_preview_var.set(" ".join(command))

    @staticmethod
    def _resolve_literature_paths(paths: list[str]) -> tuple[str | None, str | None]:
        literature_cards = None
        rag_index = None
        for raw in paths:
            path = Path(raw)
            lower = path.name.lower()
            if lower.endswith(".jsonl") and rag_index is None:
                rag_index = str(path)
            elif lower.endswith(".csv") and literature_cards is None:
                literature_cards = str(path)
        return literature_cards, rag_index


class HypothesisSelectionDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, candidates: dict[str, dict[str, object]]) -> None:
        super().__init__(parent)
        self.title("Select Hypothesis")
        self.geometry("1100x720")
        self.transient(parent)
        self.grab_set()
        self.candidates = candidates
        self.selection: DeepDiveSelection | None = None
        self.selected_id_var = tk.StringVar()
        self.title_var = tk.StringVar()

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        self.candidate_list = tk.Listbox(left, width=34, exportselection=False)
        self.candidate_list.grid(row=0, column=0, sticky="nsew")
        for hyp_id, candidate in sorted(candidates.items()):
            self.candidate_list.insert(tk.END, f"{hyp_id}: {candidate.get('title', '')}")
        self.candidate_list.bind("<<ListboxSelect>>", self._load_selected_candidate)

        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.rowconfigure(3, weight=1)
        right.rowconfigure(4, weight=1)
        right.rowconfigure(5, weight=1)
        right.rowconfigure(6, weight=1)

        ttk.Label(right, text="Hypothesis ID").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(right, textvariable=self.selected_id_var, state="readonly").grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(right, text="Title").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(right, textvariable=self.title_var).grid(row=1, column=1, sticky="ew", pady=4)
        self.statement_text = self._text_field(right, "Hypothesis statement", 2)
        self.explanation_text = self._text_field(right, "Explanation", 3)
        self.tests_text = self._text_field(right, "Required tests", 4)
        self.falsify_text = self._text_field(right, "Falsification criteria", 5)
        self.sources_text = self._text_field(right, "Source tables", 6)

        buttons = ttk.Frame(right)
        buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Use Selected Hypothesis", command=self._confirm).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=self._cancel).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        if self.candidate_list.size():
            self.candidate_list.selection_set(0)
            self._load_selected_candidate()

    def _text_field(self, parent: ttk.Frame, label: str, row: int) -> tk.Text:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", pady=4)
        text = tk.Text(parent, height=5, wrap="word")
        text.grid(row=row, column=1, sticky="nsew", pady=4)
        return text

    def _load_selected_candidate(self, event: object | None = None) -> None:
        selection = self.candidate_list.curselection()
        if not selection:
            return
        label = self.candidate_list.get(selection[0])
        hyp_id = label.split(":", 1)[0].strip()
        candidate = self.candidates[hyp_id]
        raw_candidate = candidate.get("raw") if isinstance(candidate.get("raw"), dict) else {}
        self.selected_id_var.set(hyp_id)
        self.title_var.set(str(candidate.get("title") or hyp_id))
        self._set_text(self.statement_text, str(candidate.get("hypothesis_statement") or ""))
        self._set_text(self.explanation_text, str(candidate.get("plain_language_explanation") or ""))
        self._set_text(self.tests_text, _list_to_text(raw_candidate.get("key_validation")))
        self._set_text(self.falsify_text, _list_to_text(raw_candidate.get("falsification_criteria")))
        self._set_text(self.sources_text, _list_to_text(raw_candidate.get("required_output_tables")))

    def _confirm(self) -> None:
        hyp_id = self.selected_id_var.get().strip()
        if not hyp_id or hyp_id not in self.candidates:
            messagebox.showerror("No hypothesis selected", "Select a hypothesis before continuing.")
            return
        candidate = self.candidates[hyp_id]
        self.selection = DeepDiveSelection(
            hypothesis_id=hyp_id,
            title=self.title_var.get().strip() or hyp_id,
            selected_hypothesis=self._get_text(self.statement_text),
            plain_language_explanation=self._get_text(self.explanation_text),
            rationale="Selected and edited through the GUI launcher.",
            required_tests=_text_to_list(self._get_text(self.tests_text)),
            falsification_criteria=_text_to_list(self._get_text(self.falsify_text)),
            source_tables=_text_to_list(self._get_text(self.sources_text)) or ["rag_grounded_hypothesis_candidates.md"],
            selected_candidate_source="gui_hypothesis_selection",
            selected_candidate_text=str(candidate.get("source_text") or "").strip(),
            selection_mode="gui_candidate_selection_for_deep_dive",
            data_support_level="not_assessed",
        )
        self.destroy()

    def _cancel(self) -> None:
        self.selection = None
        self.destroy()

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)

    @staticmethod
    def _get_text(widget: tk.Text) -> str:
        return widget.get("1.0", tk.END).strip()


def _list_to_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    if value:
        return str(value)
    return ""


def _text_to_list(value: str) -> list[str]:
    return [line.strip("- ").strip() for line in value.splitlines() if line.strip("- ").strip()]


def main() -> int:
    app = ScRTAgentLauncher()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
