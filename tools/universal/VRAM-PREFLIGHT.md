# Before you run

Close GPU apps, browsers and overlays. On a 16GB card, try the 12GB preset;
otherwise use a separately validated smaller-context preset.

The launcher reads free VRAM before loading weights, reserves room for a
minimized desktop, and chooses the largest validated arena that fits. If none
fits, it reports free and required MiB and exits before starting the server.
The boot output explains the selected arena. A warning lists compute processes
where the driver exposes them; Windows WDDM may expose only aggregate memory
use, and graphics processes are not necessarily included in the compute list.

`--arena-mib` selects a measured arena explicitly. It cannot select a new arena
or bypass the memory check. `--desktop-reserve-mib` overrides the desktop
reserve, while the fixed safety margin still applies. Leave the shipped reserve
in place unless you have measured your setup. Browsers, GPU wallpapers and game
overlays are outside the minimized-desktop allowance.

# Sealed profile data

Each profile must include `vram_preflight` inside its integrity-protected JSON:

```json
{
  "desktop_reserve_mib": 640,
  "safety_margin_mib": 64,
  "validated_arenas": [
    {
      "arena_mib": 2048,
      "required_free_mib": 11400,
      "evidence": "path to an architecture/profile-matched receipt"
    }
  ]
}
```

These are **synthetic schema examples**, not the selected release reserve,
measured requirements or certification evidence. Phase 1 must supply actual
architecture-specific measurements, sources and the reserve decision. Legacy
profiles without these data refuse to start through the updated launcher.

`required_free_mib` is the measured incremental requirement for the complete
profile, including context capacity, vision, drafter and lifecycle peaks. When
only an absolute total peak is known, that peak is a conservative requirement;
do not subtract an assumed baseline. Any baseline subtraction needs a measured
matching receipt. The safety margin is added to the measured requirement.

The usable budget is `min(driver_free_mib, total_mib - desktop_reserve_mib)`.
This avoids subtracting existing desktop allocations twice. The launcher selects
only entries whose requirement plus safety margin fits that budget; arena
changes affect only the existing `--shared-device-memory-mib` argument. Every
entry therefore needs evidence for the same binary, numerical path, context,
batch, host cache and remaining environment. If a fallback needs other settings,
it must be a separately sealed preset rather than an entry in this list.

Run 300 includes unit and native launcher integration checks. It does **not**
prove the required dummy-allocation tests or physical 12GB Windows compatibility.
It also records both `nvidia-smi` and CUDA memory readings: these differ under
WDDM, so reported free MiB alone is not proof that a physical SKU will fit.
