package com.yuyan.imemodule.llm.scheduler

import java.util.concurrent.atomic.AtomicReference

class ModelMutexScheduler {

    enum class State {
        Idle,
        Inference,
        MemoryProcessing,
    }

    private val stateRef = AtomicReference(State.Idle)
    private val runningMemoryToken = AtomicReference<String?>(null)

    fun state(): State = stateRef.get()

    fun preemptForInference(): String? {
        val token = runningMemoryToken.getAndSet(null)
        stateRef.set(State.Inference)
        return token
    }

    fun markInferenceFinished() {
        stateRef.compareAndSet(State.Inference, State.Idle)
    }

    fun tryEnterMemoryProcessing(): String? {
        if (!stateRef.compareAndSet(State.Idle, State.MemoryProcessing)) return null
        val token = "mem-" + System.nanoTime().toString()
        runningMemoryToken.set(token)
        return token
    }

    fun isMemoryTokenValid(token: String): Boolean {
        return runningMemoryToken.get() == token && stateRef.get() == State.MemoryProcessing
    }

    fun markMemoryFinished(token: String) {
        if (runningMemoryToken.compareAndSet(token, null)) {
            stateRef.compareAndSet(State.MemoryProcessing, State.Idle)
        }
    }
}
