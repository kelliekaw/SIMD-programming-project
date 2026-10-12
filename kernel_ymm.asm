default rel
section .text

global kernel_ymm

; void kernel_ymm(
;     const float *A,  ; RCX
;     const float *X,  ; RDX
;     float *Y,        ; R8
;     size_t m,        ; R9
;     size_t n         ; [RSP+40] on entry
; );

kernel_ymm:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15

    mov rbx, rcx        ; current matrix row
    mov rsi, rdx        ; X base address
    mov rdi, r8         ; Y base address
    mov r12, r9         ; m
    mov r13, [rsp+96]   ; n

    xor r14, r14        ; i = 0

.outer_loop:
    cmp r14, r12
    jae .done

    xor r15, r15        ; j = 0
    vxorps ymm0, ymm0, ymm0  ; eight partial sums = 0

.vector_loop:
    ; Continue only if at least eight elements remain
    mov rax, r13
    sub rax, r15
    cmp rax, 8
    jb .reduce

    vmovups ymm1, [rbx+r15*4]
    vmovups ymm2, [rsi+r15*4]

    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    add r15, 8
    jmp .vector_loop

.reduce:
    ; Add the upper four lanes to the lower four lanes
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1

    ; Reduce the four remaining lanes to one scalar
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

.scalar_tail:
    cmp r15, r13
    jae .store_result

    vmovss xmm1, [rbx+r15*4]
    vmulss xmm1, xmm1, [rsi+r15*4]
    vaddss xmm0, xmm0, xmm1

    inc r15
    jmp .scalar_tail

.store_result:
    vmovss [rdi+r14*4], xmm0

    lea rax, [r13*4]
    add rbx, rax

    inc r14
    jmp .outer_loop

.done:
    ; Clear upper YMM state before returning
    vzeroupper

    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret