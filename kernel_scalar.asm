
default rel
section .text

; Columns per block. MUST match BLOCK_SIZE in main.c.
%define BLOCK_SIZE 4096

global kernel_scalar

; void kernel_scalar(
;     const float *A,  ; RCX
;     const float *X,  ; RDX
;     float *Y,        ; R8
;     size_t m,        ; R9
;     size_t n         ; [RSP+40] on entry
; );

kernel_scalar:
    ; Preserve Windows x64 nonvolatile registers
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15

    ; Seven pushes = 56 bytes.
    ; The fifth argument is now at [rsp+96].
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
    mov r13, [rsp+96]

    xor r14, r14          ; i = 0

.outer_loop:
    cmp r14, r12
    jae .done

    xor r15, r15          ; j = 0
    xorps xmm3, xmm3      ; row total = 0.0

.block_loop:
    cmp r15, r13
    jae .store_result

    ; r10 = min(j + BLOCK_SIZE, n) = end of this block
    lea r10, [r15+BLOCK_SIZE]
    cmp r10, r13
    cmova r10, r13

    xorps xmm0, xmm0      ; block partial sum = 0.0

.inner_loop:
    cmp r15, r10
    jae .end_block

    movss xmm1, [rbx+r15*4]
    mulss xmm1, [rsi+r15*4]
    addss xmm0, xmm1

    inc r15
    jmp .inner_loop

.end_block:
    addss xmm3, xmm0      ; row total += block partial
    jmp .block_loop

.store_result:
    movss [rdi+r14*4], xmm3

    ; Move A to the next row: n * sizeof(float)
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