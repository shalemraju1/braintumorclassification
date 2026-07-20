(() => {
    const body = document.body;

    const animateProgressBars = () => {
        document.querySelectorAll(".progress-fill[data-progress]").forEach((bar) => {
            const value = Number(bar.dataset.progress || 0);
            requestAnimationFrame(() => {
                bar.style.width = `${Math.max(0, Math.min(value, 100))}%`;
            });
        });
    };

    const initDropZones = () => {
        document.querySelectorAll("[data-dropzone]").forEach((zone) => {
            const input = zone.querySelector("input[type='file']");
            const preview = zone.querySelector("[data-preview-image]");
            const hint = zone.querySelector("[data-file-name]");

            const setPreview = (file) => {
                if (!file) {
                    return;
                }

                const reader = new FileReader();
                reader.onload = () => {
                    if (preview) {
                        preview.src = reader.result;
                        preview.classList.add("is-visible");
                    }
                };
                reader.readAsDataURL(file);

                if (hint) {
                    hint.textContent = file.name;
                }
            };

            if (input) {
                input.addEventListener("change", () => {
                    setPreview(input.files && input.files[0]);
                });
            }

            zone.addEventListener("dragover", (event) => {
                event.preventDefault();
                zone.classList.add("is-active");
            });

            zone.addEventListener("dragleave", () => {
                zone.classList.remove("is-active");
            });

            zone.addEventListener("drop", (event) => {
                event.preventDefault();
                zone.classList.remove("is-active");

                const file = event.dataTransfer.files && event.dataTransfer.files[0];
                if (!file || !input) {
                    return;
                }

                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                input.files = dataTransfer.files;
                setPreview(file);
            });
        });
    };

    const initModals = () => {
        const overlay = document.querySelector("[data-modal-overlay]");
        const modalImage = overlay ? overlay.querySelector("[data-modal-image]") : null;
        const modalTitle = overlay ? overlay.querySelector("[data-modal-title]") : null;
        const closeButtons = document.querySelectorAll("[data-modal-close]");

        const openModal = (src, title) => {
            if (!overlay || !modalImage) {
                return;
            }

            modalImage.src = src;
            if (modalTitle) {
                modalTitle.textContent = title || "Preview";
            }
            overlay.classList.add("is-open");
        };

        const closeModal = () => {
            if (!overlay) {
                return;
            }
            overlay.classList.remove("is-open");
            if (modalImage) {
                modalImage.src = "";
            }
        };

        document.querySelectorAll("[data-preview-trigger]").forEach((trigger) => {
            trigger.addEventListener("click", () => {
                openModal(trigger.dataset.previewSrc, trigger.dataset.previewTitle);
            });
        });

        closeButtons.forEach((button) => button.addEventListener("click", closeModal));

        if (overlay) {
            overlay.addEventListener("click", (event) => {
                if (event.target === overlay) {
                    closeModal();
                }
            });
        }

        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                closeModal();
            }
        });
    };

    const initConfirmations = () => {
        document.querySelectorAll("[data-confirm]").forEach((button) => {
            button.addEventListener("click", (event) => {
                const message = button.dataset.confirm;
                if (message && !window.confirm(message)) {
                    event.preventDefault();
                }
            });
        });
    };

    const initLoadingState = () => {
        const overlay = document.querySelector("[data-loading-overlay]");
        if (!overlay) {
            return;
        }

        document.querySelectorAll("form[data-loading-form]").forEach((form) => {
            form.addEventListener("submit", () => {
                overlay.classList.add("is-visible");
            });
        });
    };

    const initFlashDismiss = () => {
        document.querySelectorAll("[data-flash-dismiss]").forEach((button) => {
            button.addEventListener("click", () => {
                const flash = button.closest(".flash");
                if (flash) {
                    flash.remove();
                }
            });
        });
    };

    const initBodyTransition = () => {
        body.classList.add("page-enter");
        window.setTimeout(() => body.classList.remove("page-enter"), 500);
    };

    animateProgressBars();
    initDropZones();
    initModals();
    initConfirmations();
    initLoadingState();
    initFlashDismiss();
    initBodyTransition();
})();
