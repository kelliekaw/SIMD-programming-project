default rel
section .text

global kernel_xmm

; void kernel_xmm(
;     const float *A,  ; RCX
;     const float *X,  ; RDX
;     float *Y,        ; R8
;     size_t m,        ; R9
;     size_t n         ; [RSP+40] on entry
; );

kernel_xmm:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15

    ; Seven pushes move the fifth argument to [rsp+96]
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
    vxorps xmm0, xmm0, xmm0  ; four partial sums = 0

.vector_loop:
    ; Continue only if at least four elements remain
    mov rax, r13
    sub rax, r15
    cmp rax, 4
    jb .reduce

    ; Load four A values and four X values
    vmovups xmm1, [rbx+r15*4]
    vmovups xmm2, [rsi+r15*4]

    ; Four multiplications and four additions
    vmulps xmm1, xmm1, xmm2
    vaddps xmm0, xmm0, xmm1

    add r15, 4
    jmp .vector_loop

.reduce:
    ; Add the four accumulated lanes together
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

.scalar_tail:
    ; Process leftover elements, if any
    cmp r15, r13
    jae .store_result

    vmovss xmm1, [rbx+r15*4]
    vmulss xmm1, xmm1, [rsi+r15*4]
    vaddss xmm0, xmm0, xmm1

    inc r15
    jmp .scalar_tail

.store_result:
    vmovss [rdi+r14*4], xmm0

    ; Advance to the next matrix row
    lea rax, [r13*4]
    add rbx, rax

    inc r14
    jmp .outer_loop

.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret